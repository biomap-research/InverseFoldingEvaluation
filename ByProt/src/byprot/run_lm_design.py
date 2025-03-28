"""Design and score using LM-Design"""
import os
import sys
import torch
import argparse
import pandas as pd

from pytorch_lightning import seed_everything
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List

cwd = os.path.dirname(__file__)
sys.path.append("%s/.." % cwd)

from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer, GenOut
from byprot import utils


def score_by_categorical(
    logits: torch.Tensor,   # [B, seq_len, vocab_size],
    tokens: torch.Tensor, #[B, seq_len],
    temperature: float,
):
    dist = torch.distributions.Categorical(logits=logits.div(temperature))
    scores = dist.log_prob(tokens)
    return scores

def inpaint(
    pdb_path: str, 
    load_chains: List[str],
    designer: Designer, 
    design_range: Dict[str, Tuple], 
    num_generate: int, 
    experiment_path: str,
    need_attn_weights=False,
    temperature=0.1,
    strategy="denoise", # llm default strategy is denoise
    max_iter=1,
    score_only=False
    ):
    """Inpaint design for.

    Args:
        pdb_path: pdb file path
        load_chains: List[str], chain ids to extract coords.
        design_range: Dict[str, Tuple], chain and design range.
        num_generate: int, generate per input pdb.
    """
    designer.set_structure(pdb_path, chain_list=load_chains)
    structure = designer._structure
    # Designer.set_structure
    # pdb_path: str pdb file path
    # chain_list: List, chain list to extract info
    # masked_chain_list: List = None, chains to design
    batch = designer.alphabet.featurize(raw_batch=[designer._structure])
    if designer.cfg.cuda:
        batch = utils.recursive_to(batch, designer._device)

    start_ids = []
    end_ids = []
    chain_order = []
    prev_length = 0
    for (chain, length) in zip(batch["chain_order"][0], batch["chain_lengths"][0]):
        if chain in design_range:
            s, e = design_range[chain]
            if e == -1:
                e = length
            start_ids.append(prev_length + s)
            end_ids.append(prev_length + e)
            chain_order.append(chain)
        prev_length += length
    prev_tokens = batch['tokens'].clone()
    for sid, eid in zip(start_ids, end_ids):
        prev_tokens[..., sid:eid+1] = designer.alphabet.mask_idx

    batch['prev_tokens'] = prev_tokens
    batch['prev_token_mask'] = prev_tokens.eq(designer.alphabet.mask_idx) 

    native_chains = {}
    native_tokens = batch['tokens'].clone().detach()
    for sid, eid, chain in zip(start_ids, end_ids, chain_order): 
        original_segment = designer.alphabet.decode(
            native_tokens[..., sid:eid+1], 
            remove_special=False
        )
        if chain not in native_chains:
            native_chains[chain] = original_segment[0]


    designed_chains = {}
    gen_outs = []
    designed_chain_tokens = {}
    designed_seq_score = {}
    native_seq_score = {}
    for _ in range(num_generate):
        outputs = designer.generator.generate(
            model=designer.model, 
            batch=batch,
            need_attn_weights=need_attn_weights,
            replace_visible_tokens=True,
            temperature=temperature,
            max_iter=max_iter,
            strategy=strategy,
        )
        output_tokens = outputs[0]
        if _ == 0:
            output_logits = outputs[2].clone().detach() # [B, seq_len, vocab_size]
            native_scores = score_by_categorical(output_logits, native_tokens, temperature).to("cpu")
            for sid, eid, chain in zip(start_ids, end_ids, chain_order):
                log_probs_native_chain = native_scores[..., sid:eid+1]   # log probs of the output tokens
                native_seq_score[chain] = (-torch.sum(log_probs_native_chain[0], dim=-1) / log_probs_native_chain.size(-1)).item()
            if score_only:
                break

        for sid, eid, chain in zip(start_ids, end_ids, chain_order):
            designed_tokens_i = output_tokens[..., sid:eid+1].clone().detach().to("cpu")
            designed_segment_i = designer.alphabet.decode(
                designed_tokens_i, 
                remove_special=False
            )
            log_probs_i = outputs[1][..., sid:eid+1].clone().detach().to("cpu")   # log probs of the output tokens
            if chain not in designed_chains:
                designed_chains[chain] = designed_segment_i
                designed_seq_score[chain] = [(-torch.sum(log_probs_i[0], dim=-1) / log_probs_i.size(-1)).item()]
                designed_chain_tokens[chain] = [designed_tokens_i]
            else:
                designed_chains[chain].append(designed_segment_i[0])
                designed_seq_score[chain].append(
                    (-torch.sum(log_probs_i[0], dim=-1) / log_probs_i.size(-1)).item()
                )
                designed_chain_tokens[chain].append(designed_tokens_i)

        

        output_tokens = designer.alphabet.decode(output_tokens, remove_special=True)
        gen_outs.append(
            {
                "output_tokens": output_tokens,
                "output_scores": outputs[1].detach().to("cpu"),
                "attentions": outputs[2].detach().to("cpu") if need_attn_weights else None
            }
        )
    del batch
        
    params = {
        "start_ids": start_ids,
        "end_ids": end_ids,
        "chain_order": chain_order,
    }
    return native_chains, native_seq_score, designed_chains, designed_seq_score, gen_outs, params

def main(
    csv_path: str,
    save_dir: str,
    pdb_dir: str,
    condition_column: str,  # condition on which chain
    target_column: str, # which column to design
    experiment_path: str,
    random: int = 0,
    temperature: float = 0.2,
    seq_gen_num: int = 100,
    score_only: bool = False,
):
    print("seed", random)
    print("temperature", temperature)
    seed_everything(random)
    cfg = Cfg(
        cuda=True,
        generator=Cfg(
            max_iter=1,
            strategy='mask_predict',
            temperature=temperature
        )
    )
    save_dir = Path(save_dir)
    csv_path = Path(csv_path)
    pdb_dir = Path(pdb_dir)

    tag = f"{csv_path.stem}_seed={random}_tempr={temperature}_condition={condition_column.replace(',', '|') if condition_column != '' else None}"
    save_dir = save_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    info_dir = save_dir / "generate_info"
    info_dir.mkdir(parents=True, exist_ok=True)

    designer = Designer(experiment_path=experiment_path, cfg=cfg)

    df = pd.read_csv(csv_path)

    condition_column = condition_column.split(",") if condition_column != "" else []

    target_column = target_column.split(",")

    seq_dfs = []
    for i,r in tqdm(df.iterrows()):
        name = r["pdb_name"]
        pdb = str(pdb_dir / f"{name}.pdb")
        load_chains = []
        target_chains = []
        design_range = {}
        for c in target_column:
            for chain in r[c]:
                load_chains.append(chain)
                target_chains.append(chain)
                design_range[chain] = (1, -1)
        
        for c in condition_column:
            if c == "ALL":
                load_chains = []    # condition on all chains except target chains
                break
            load_chains.extend([x for x in r[c]])
        
        info_dir_i = info_dir / name
        info_dir_i.mkdir(parents=True, exist_ok=True)

        native_chains, native_scores, designed_chains, designed_scores, gen_outs, params = inpaint(
            pdb,
            load_chains, 
            designer, 
            design_range, 
            num_generate=seq_gen_num, 
            need_attn_weights=False,
            temperature=temperature,
            strategy="denoise",
            max_iter=1,
            score_only=score_only
        )

        dfc = pd.DataFrame([native_chains])
        dfc["pdb_name"] = name
        dfs = pd.DataFrame([native_scores])
        dfs["pdb_name"] = name
        native_df = pd.merge(dfc, dfs, on="pdb_name", how="left", suffixes=("_native_seq", "_native_score"))
        if not score_only:
            df = pd.DataFrame(designed_chains)
            df["pdb_name"] = name
            df["pdb_path"] = pdb
            
            seq_dfs.append(pd.merge(df, native_df, on="pdb_name", jow="left"))
            if not (info_dir_i / f"genouts.pth").exists():
                torch.save(gen_outs, str(info_dir_i / f"genouts.pth"))
        else:
            native_df["pdb_path"] = pdb
            seq_dfs.append(native_df)
            
        if not (info_dir_i / f"chain_ranges.pth").exists():
            torch.save(params, str(info_dir_i / f"chain_ranges.pth"))
        
    seq_dfs = pd.concat(seq_dfs)
    seq_dfs.to_csv(save_dir / f"LM-Design-Results.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run lm design")
    parser.add_argument('--csv_path', type=str, help='csv path, must contains "pdb_name"')
    parser.add_argument('--pdb_dir', type=str, help='dir where the input pdb files saved, file name must same with pdb_name in csv_path')
    parser.add_argument('--experiment_path', type=str, help='dir to lm_design_esm2_650m')
    parser.add_argument('--save_dir', type=str, help='save dir')
    parser.add_argument('-s', type=int, help='ramdom seed', default=0)
    parser.add_argument('-n', type=int, help='seqs generate per sequence', default=100)
    parser.add_argument('-t', type=float, help='temperature', default=0.2)
    parser.add_argument('--condition_column', type=str, help='"" for no conditions "ALL" for condition on all except target chains, seperated by ","', default="")
    parser.add_argument('--target_column', type=str, help='chains to design, seperated by ","', default="vh_chain,vl_chain")
    parser.add_argument('--score_only', action="store_true", help="score native sequence only")

    args = parser.parse_args()

    main(
        args.csv_path,
        args.save_dir,
        args.pdb_dir,
        args.condition_column,
        args.target_column,
        args.experiment_path,
        args.s,
        args.t,
        args.n,
        args.score_only
    )
