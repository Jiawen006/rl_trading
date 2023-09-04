# About Dataset

The dataset consists of 10 anonymized financial time series. Each time series includes the open price, high price, low price, close price and trading volume for each day.

## Description of the data

The entire dataset is divided into training dataset and test dataset. The training dataset consists of financial time series for 2088 days before the start of trading. The test dataset, also called the trading dataset in this problem, can be used to evaluate the effectiveness of the method.

Within each dataset, files are stored according to different stocks. For example, `01.csv` represents the first stock in the folder.

```
Dataset Structure
Root Dir/
  -train/
    - 01.csv
    - 02.csv
    -...
  -test/
    - 01.csv
    - 02.csv
    -...
  -README.md

```

### File formats

All data are stored in `csv` files.

```
-20 files, format csv.
```

## Authors

* **[Dr. John Fearnley](john.fearnley@liverpool.ac.uk)** - *Initial work*


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
