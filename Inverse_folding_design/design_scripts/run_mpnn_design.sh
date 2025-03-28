PROJECT_HOME=$(cd "$(dirname "$0")" && pwd)/..

mpnn_home=$1 # mpnn repo dir
weight_dir=$2 # parent dir of mpnn checkpoint files, like v_48_002.pt, abmpnn.pt ...


vhh_csv_path=${PROJECT_HOME}/input_pdbs/df_vhh_23.csv
fab_csv_path=${PROJECT_HOME}/input_pdbs/df_fab_23.csv

vhh_pdb_dir=${PROJECT_HOME}/input_pdbs/vhh
fab_pdb_dir=${PROJECT_HOME}/input_pdbs/fab

out_dir=${PROJECT_HOME}/mpnn_designs

mkdir -p ${out_dir}

python ${PROJECT_HOME}/model_interface/run_mpnn.py \
    --csv_path ${vhh_csv_path} \
    --pdb_dir ${vhh_pdb_dir} \
    --weight_dir ${weight_dir} \
    --mpnn_home ${mpnn_home} \
    --model_name abmpnn \
    --out_dir ${out_dir}/abmpnn-vhh-fullseq

python ${PROJECT_HOME}/model_interface/run_mpnn.py \
    --csv_path ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --weight_dir ${weight_dir} \
    --mpnn_home ${mpnn_home} \
    --model_name abmpnn \
    --out_dir ${out_dir}/abmpnn-fab-fullseq

python ${PROJECT_HOME}/model_interface/run_mpnn.py \
    --csv_path ${vhh_csv_path} \
    --pdb_dir ${vhh_pdb_dir} \
    --weight_dir ${weight_dir} \
    --mpnn_home ${mpnn_home} \
    --model_name v_48_020 \
    --out_dir ${out_dir}/mpnn-vhh-fullseq

python ${PROJECT_HOME}/model_interface/run_mpnn.py \
    --csv_path ${fab_csv_path} \
    --pdb_dir ${fab_pdb_dir} \
    --weight_dir ${weight_dir} \
    --mpnn_home ${mpnn_home} \
    --model_name v_48_020 \
    --out_dir ${out_dir}/mpnn-fab-fullseq