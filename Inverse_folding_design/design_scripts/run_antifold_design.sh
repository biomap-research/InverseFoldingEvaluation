PROJECT_HOME=$(cd "$(dirname "$0")" && pwd)/..

antifold_home=$1 # antifold repo dir


vhh_csv_path=${PROJECT_HOME}/input_pdbs/df_vhh_23.csv
fab_csv_path=${PROJECT_HOME}/input_pdbs/df_fab_23.csv

vhh_pdb_dir=${PROJECT_HOME}/input_pdbs/vhh
fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab

out_dir=${PROJECT_HOME}/antifold_designs

mkdir -p ${out_dir}

python ${antifold_home}/antifold/main.py \
    --pdb_dir ${vhh_pdb_dir} \
    --pdbs_csv ${vhh_csv_path} \
    --num_seq_per_target 100 \
    --regions "all" \
    --sampling_temp "0.2" \
    --out_dir ${out_dir}/antifold-vhh-fullseq

python ${antifold_home}/antifold/main.py \
    --pdb_dir ${fab_pdb_dir} \
    --pdbs_csv ${fab_csv_path} \
    --num_seq_per_target 100 \
    --regions "all" \
    --sampling_temp "0.2" \
    --out_dir ${out_dir}/antifold-fab-fullseq

fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab_apo
python ${antifold_home}/antifold/main.py \
    --pdb_dir ${fab_pdb_dir} \
    --pdbs_csv ${fab_csv_path} \
    --num_seq_per_target 100 \
    --regions "all" \
    --sampling_temp "0.2" \
    --out_dir ${out_dir}/antifold_noAG-fab-fullseq

fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab_relaxed
python ${antifold_home}/antifold/main.py \
    --pdb_dir ${fab_pdb_dir} \
    --pdbs_csv ${fab_csv_path} \
    --num_seq_per_target 100 \
    --regions "all" \
    --sampling_temp "0.2" \
    --out_dir ${out_dir}/antifold_relaxed-fab-fullseq