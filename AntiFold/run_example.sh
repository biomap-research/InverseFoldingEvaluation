#!/bin/bash
OUTPUT_DIRECTORY="output/single_pdb/"
UPLOADED_FILE="data/pdbs/6y1l_imgt.pdb"
HEAVY_CHAIN="H"
LIGHT_CHAIN="L"
SAMPLE_NUMBER=1
SAMPLING_TEMP=0.20
REGIONS="all"
LIMIT_VARIATION_STRING=""

python antifold/main.py \
--out_dir "$OUTPUT_DIRECTORY" \
--pdb_file "$UPLOADED_FILE" \
--heavy_chain "$HEAVY_CHAIN" \
--light_chain "$LIGHT_CHAIN" \
--regions "$REGIONS" \
--num_seq_per_target "$SAMPLE_NUMBER" \
--sampling_temp "$SAMPLING_TEMP" \
$LIMIT_VARIATION_STRING

# Output
# "$OUTPUT_DIRECTORY"/log.txt
# "$OUTPUT_DIRECTORY"/6y1l_imgt.fasta
# "$OUTPUT_DIRECTORY"/6y1l_imgt.csv