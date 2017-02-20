import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
from scipy.stats import ttest_ind

def read_data():
    df_twin_cities = pd.read_excel('~/Downloads/AKQA_Dataset_Test.xlsx',
                                   sheetname = 'Twin Cities')
    df_zips = pd.read_excel('~/Downloads/AKQA_Dataset_Test.xlsx',
                            sheetname = 'Zips')
    return df_twin_cities, df_zips

def eda(df):
    # 211 unique real estate company names
    print len(df.Realty.unique())

def add_parent_company(df):
    s = df.Realty.str.replace(',', '')
    s = s.str.replace('/', ' ')
    s = s.str.replace('.', ' ')
    s = s.str.split().apply(lambda x: x[0] + ' ' + x[1])
    df['ParentCompany'] = s
    return df

def plot_comparison(df1, df2, column, label, bins):
    # Plot Square Footage comparison
    col1 = df1[column]
    col1 = col1[~np.isnan(col1)]

    col2 = df2[column]
    col2 = col2[~np.isnan(col2)]

    plt.hist(col1,
             bins = bins,
             normed = True,
             fc = (0, 0, 1, 0.5),
             label = 'Short Sale')
    plt.hist(col2,
             bins = bins,
             normed = True,
             fc=(1, 0, 0, 0.5),
             label = 'Non Short Sale')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title('Short Sale vs. Non Short Sale ' + label)
    plt.legend()
    plt.show()

def ttest_two_mean(df1, df2, column):
    # Plot Square Footage comparison
    col1 = df1[column]
    col1 = col1[~np.isnan(col1)]

    col2 = df2[column]
    col2 = col2[~np.isnan(col2)]

    results = ttest_ind(col1, col2, equal_var = False)
    print results

    return results

def part_1(df):
    # Add column for parent company
    df = add_parent_company(df)

    # Group by parent real estate company
    print df.groupby('ParentCompany').size().sort_values()[-6:]

    # Plot by number of houses
    df.groupby('ParentCompany').size().sort_values()[-6:].plot(kind = 'bar')

    # Label Axes
    plt.xticks(rotation = 30, horizontalalignment = 'right')
    plt.xlabel('Company')
    plt.ylabel('Number of Houses for Sale')
    plt.title('Houses for Sale by Company')
    plt.show()

def part_2(df):
    # Scatter data points
    df[['ListPrice', 'SQFT']].sort_values('SQFT').plot(x = 'SQFT',
                                                       y = 'ListPrice',
                                                       kind = 'scatter')

    # Linear regression
    x = df.SQFT.values
    y = df.ListPrice.values
    X = np.column_stack((x, x**2))
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X)
    results = model.fit()

    # Plot regression line
    x = np.array(range(10000))
    y = results.params[0] + x * results.params[1] + x**2 * results.params[2]
    plt.plot(x, y)

    # Plot aesthetics
    plt.xlabel('Square Footage')
    plt.ylabel('List Price')
    plt.title('List Price vs. Square Footage')
    plt.xlim([0,10000])
    plt.ylim([0,6000000])
    plt.show()

    # Predict price for 2111 square foot house
    x = 2111
    price = results.params[0] + x * results.params[1] + x**2 * results.params[2]
    print "Price for 2111 square foot house should be: {}".format(price)

def part_3(df, show_plots = True):
    # Isolate zip code data
    df_55104 = df[df.ZIP == 55104].ListPrice
    df_55108 = df[df.ZIP == 55108].ListPrice

    # Plot 55108 area code
    df_55108.plot(kind = 'hist')
    plt.xlabel('List Price')
    plt.title('55108 List Price Distribution')
    if show_plots:
        plt.show()

    # Plot 55104 area code
    df_55104.plot(kind = 'hist')
    plt.xlabel('List Price')
    plt.title('55104 List Price Distribution')
    if show_plots:
        plt.show()

    # Get middle 50%
    quartiles_55104 = df_55104.quantile(q = [.25, .75])
    quartiles_55108 = df_55108.quantile(q = [.25, .75])

    # Plot 55108 middle 50%
    df_55108_middle = df_55108[(df_55108 > quartiles_55108[0.25]) & (df_55108 <= quartiles_55108[0.75])]
    df_55108_middle.plot(kind = 'hist')
    plt.xlabel('List Price')
    plt.title('Middle 50% 55108 List Price Distribution')
    if show_plots:
        plt.show()

    # Plot 55104 middle 50%
    df_55104_middle = df_55104[(df_55104 > quartiles_55104[0.25]) & (df_55104 <= quartiles_55104[0.75])]
    df_55104_middle.plot(kind = 'hist')
    plt.xlabel('List Price')
    plt.title('Middle 50% 55104 List Price Distribution')
    if show_plots:
        plt.show()

    print df_55104_middle.head()

    # Compare Means
    print ttest_ind(df_55104_middle, df_55108_middle, equal_var = False)

def part_4(df, show_plots = True):
    # Scatter data points
    df[['SQFT', 'ListPrice']].plot(x = 'SQFT',
                                   y = 'ListPrice',
                                   kind = 'scatter')
    plt.xlabel('Square Feet')
    plt.ylabel('List Price')
    plt.title('List Price vs. Square Feet')
    plt.xlim([0,10000])
    plt.ylim([0,5000000])
    if show_plots:
        plt.show()

    df[['BEDS', 'ListPrice']].plot(x = 'BEDS',
                                   y = 'ListPrice',
                                   kind = 'scatter')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('List Price')
    plt.title('List Price vs. Number of Bedrooms')
    plt.xlim([0,10])
    plt.ylim([0,5000000])
    if show_plots:
        plt.show()

    df[['LotSize', 'ListPrice']].plot(x = 'LotSize',
                                      y = 'ListPrice',
                                      kind = 'scatter')
    plt.xlabel('Lot Size')
    plt.ylabel('List Price')
    plt.title('List Price vs. Lot Size')
    plt.xlim([0,140000])
    plt.ylim([0,5000000])
    if show_plots:
        plt.show()

    # Linear regression
    y = df.ListPrice
    X = np.column_stack((df.BEDS, df.SQFT, df.LotSize))
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X, missing = 'drop')
    results = model.fit()
    print results.summary()

def part_5(df, show_plots = True):
    # Add Price per Square Foot column
    df['PricePerSquareFoot'] = df.ListPrice / df.SQFT

    # Isolate into short sale and not short sale
    df_short_sale = df[df.ShortSale == 'Y']
    df_non_short_sale = df[df.ShortSale == 'N']

    # Sizes
    # print len(df_short_sale) # <- 73
    # print len(df_non_short_sale) # <- 1038

    # Make comparison plots
    if show_plots:
        plot_comparison(df_short_sale,
                        df_non_short_sale,
                        'SQFT',
                        'Square Footage',
                        np.arange(0, 9000, 500))
        plot_comparison(df_short_sale,
                        df_non_short_sale,
                        'PricePerSquareFoot',
                        'Price Per Square Foot',
                        np.arange(0, 900, 50))
        plot_comparison(df_short_sale,
                        df_non_short_sale,
                        'LotSize',
                        'Lot Size',
                        np.arange(0, 138000, 6000))

    # 2 Mean Comparison Test
    ttest_two_mean(df_short_sale, df_non_short_sale, 'SQFT')
    ttest_two_mean(df_short_sale, df_non_short_sale, 'PricePerSquareFoot')
    ttest_two_mean(df_short_sale, df_non_short_sale, 'LotSize')

    # Group by location
    bonferonni_correction = 0.05 / 17
    for location in df.LOCATION.unique():
        print location
        df_location = df[df.LOCATION == location]
        df_short_sale = df_location[df_location.ShortSale == 'Y']
        df_non_short_sale = df_location[df_location.ShortSale == 'N']
        if len(df_short_sale) > 1 and len(df_non_short_sale) > 1:
            results = ttest_two_mean(df_short_sale,
                                     df_non_short_sale,
                                     'PricePerSquareFoot')
            if results[1] < bonferonni_correction:
                print "Signifcant results"
        print "\n"

def part_6(df_twin_cities, df_zips):
    # Get counts of listings for each ZIP code
    s = df_twin_cities.groupby('ZIP').size().sort_values()
    df_zip_counts = pd.DataFrame(data = np.array([s.index, s.values]).T,
                                 columns = ['ZIP', 'Counts'])

    # Merge ZIP code counts with population data
    df_zip_counts = df_zip_counts.merge(df_zips,
                                        left_on = 'ZIP',
                                        right_on = 'ZipCode')

    # Get listings per person
    df_zip_counts['ListingsPerPerson'] = df_zip_counts.Counts / df_zip_counts.Population_2010_Census

    # Sort and display top ZIP codes
    df_zip_counts.sort(columns = 'ListingsPerPerson',
                       inplace = True,
                       ascending = False)
    print df_zip_counts.head(10)

    # Get listing price per person
    s = df_twin_cities.groupby('ZIP').mean().ListPrice
    df_list_price = pd.DataFrame(data = np.array([s.index, s.values]).T,
                                 columns = ['ZIP', 'AvgListPrice'])

    # Merge ZIP code counts with population data
    df_list_price = df_list_price.merge(df_zips,
                                        left_on = 'ZIP',
                                        right_on = 'ZipCode')

    # Get listings per person
    df_list_price['AvgListPricePerPerson'] = df_list_price.AvgListPrice / df_list_price.Population_2010_Census

    # Sort and display top ZIP codes
    df_list_price.sort(columns = 'AvgListPricePerPerson',
                       inplace = True,
                       ascending = False)
    print df_list_price.head(10)

def part_7(df):
    print df.info()

if __name__ == "__main__":
    df_twin_cities, df_zips = read_data()
    # part_1(df_twin_cities)
    # part_2(df_twin_cities)
    # part_3(df_twin_cities, show_plots = False)
    # part_4(df_twin_cities, show_plots = True)
    # part_5(df_twin_cities, show_plots = False)
    # part_6(df_twin_cities, df_zips)
    part_7(df_twin_cities)
