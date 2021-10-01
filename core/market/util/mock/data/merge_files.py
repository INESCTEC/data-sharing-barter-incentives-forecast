
with open("mock_measurements.csv") as f:
    mock_original = f.readlines()

with open("mock_measurements2.csv") as f:
    mock_new = f.readlines()

assert len(mock_original) == len(mock_new)


with open("new_measurements.csv", "w") as f:

    for i in range(len(mock_original)):
        _row_original = mock_original[i]  # remove \n
        _row_new = mock_new[i].split(';')[0]  # remove \n
        f.write(_row_new + ";" + _row_original)
