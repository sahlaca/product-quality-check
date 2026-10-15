These one-row CSVs come from the held-out test set and can be uploaded to the app.
fail_example.csv is an actual failure predicted as Fail; fail_missed_example.csv
is an actual failure predicted as Pass; pass_example.csv is an actual pass
predicted as Pass. pass_and_fail_batch.csv combines pass_example.csv and
fail_example.csv for a two-record batch demo. The app model was fitted only on
the training split. Compare predictions with the actual labels; predictions can be wrong.
