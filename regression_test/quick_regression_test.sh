#!/bin/bash

#Clean up old results
rm -f test_regression_dna_out test_regression_protein_out

#Generate new results
./program_gpu dna ./test-data/dna-reference.fasta     ./test-data/dna-query.fasta     test_regression_dna_out     >/dev/null 2>&1
./program_gpu aa  ./test-data/protein-reference.fasta ./test-data/protein-query.fasta test_regression_protein_out >/dev/null 2>&1

#Compare new results to correct results
diff_dna=$(diff test_regression_dna_out     ./regression_test/test_regression_dna_good_result    )
diff_rna=$(diff test_regression_protein_out ./regression_test/test_regression_protein_good_result)

if [ "$diff_dna" == "" ] && [ "$diff_rna" == "" ]
then
  echo "REGRESSION TESTS PASS"
  exit 0
else
  echo "#############DNA#############"
  echo "$diff_dna"
  echo "#############RNA#############"
  echo "$diff_rna"
  exit 1
fi