PROJECT_HOME=$(cd "$(dirname "$0")" && pwd)/..

esmif_home=$1 # esm repo dir


vhh_csv_path=${PROJECT_HOME}/input_pdbs/df_vhh_23.csv
fab_csv_path=${PROJECT_HOME}/input_pdbs/df_fab_23.csv

vhh_pdb_dir=${PROJECT_HOME}/input_pdbs/vhh
fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab

out_dir=${PROJECT_HOME}/esmif_designs

mkdir -p ${out_dir}

python ${esmif_home}/esm-main/examples/inverse_folding/sample_sequences_batch_fab.py \
    --input_csv ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --temperature 0.2 \
    --num-samples 100 \
    --work_dir ${out_dir}/esm_if-fab-fullseq \
    --multichain-backbone \

fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab_apo
python ${esmif_home}/esm-main/examples/inverse_folding/sample_sequences_batch_fab.py \
    --input_csv ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --temperature 0.2 \
    --num-samples 20 \
    --work_dir ${out_dir}/esm_if_noAG-fab-fullseq \
    --multichain-backbone \

fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab_relaxed
python ${esmif_home}/esm-main/examples/inverse_folding/sample_sequences_batch_fab.py \
    --input_csv ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --temperature 0.2 \
    --num-samples 20 \
    --work_dir ${out_dir}/esm_if_relaxed-fab-fullseq \
    --multichain-backbone \

python ${esmif_home}/esm-main/examples/inverse_folding/sample_sequences_batch.py \
    --input_csv ${vhh_csv_path} \
    --pdb_dir ${vhh_pdb_dir} \
    --temperature 0.2 \
    --num-samples 100 \
    --work_dir ${out_dir}/esm_if-vhh-fullseq \
    --multichain-backbone \