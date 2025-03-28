PROJECT_HOME=$(cd "$(dirname "$0")" && pwd)/..

byprot_home=$1 # byprot repo dir
experiment_path=$2 # path to lm_design_esm2_650m experiment dir

vhh_csv_path=${PROJECT_HOME}/input_pdbs/df_vhh_23.csv
fab_csv_path=${PROJECT_HOME}/input_pdbs/df_fab_23.csv

vhh_pdb_dir=${PROJECT_HOME}/input_pdbs/vhh
fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab

out_dir=${PROJECT_HOME}/byprot_lm_designs

mkdir -p ${out_dir}

python ${byprot_home}/src/byprot/run_lm_design.py \
    --csv_path ${vhh_csv_path} \
    --pdb_dir ${vhh_pdb_dir} \
    --save_dir ${out_dir}/lm_design_vanilla-vhh-fullseq \
    --experiment_path ${experiment_path} \
    --target_column "vh_chain"


python ${byprot_home}/src/byprot/run_lm_design.py \
    --csv_path ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --save_dir ${out_dir}/lm_design_vanilla-fab-fullseq \
    --experiment_path ${experiment_path} \
    --target_column "vh_chain,vl_chain"