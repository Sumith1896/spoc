To run stitching:
1. mkdir stitch_{exp_name}
2. cd stitch_{exp_name}
3. Copy the `postprocess.sh`, `cleanup.sh` and `stitch.py`
4. Copy your summary file as `{exp_name}.summary` and `input_all_test.tsv` as `{exp_name}.tsv`
5. Edit the `postprocess.sh` script to have the right number of programs to be stitched in the for loop (more is fine, process just die as out of bounds)
6. Set one of `-o`, `-x`, `-g` or `-b` to run in oracle mode, top1, gibbs & best-first mode respectively
7. Set `-p` param to the appropriate number of predictions value (usually 100)
8. If something goes wrong in the run, do `bash cleanup.sh` to remove all the generated dirs and out files.
