import pandas as pd

#read csv
riceAMLObj = pd.read_csv('Rice_MSC_Dataset.csv', sep=',')
#print shape
print("Features and attributes of dataset: ", riceAMLObj.shape)
#features we are extracting from the dataset, excluding the image color features
featureList = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ","SOLIDITY","CONVEX_AREA", "EXTENT","ASPECT_RATIO","ROUNDNESS","COMPACTNESS","SHAPEFACTOR_1","SHAPEFACTOR_2","SHAPEFACTOR_3","SHAPEFACTOR_4"]
#creating rice stats csv
f = open("riceStats.csv", "a")
#writing to file
f.write("name,max,min,mean,median,mode,standard deviation\n")
for feature in featureList:
    f.write(feature)
    f.write(",")
    f.write(str(riceAMLObj[feature].max()))
    f.write(",")
    f.write(str(riceAMLObj[feature].min()))
    f.write(",")
    f.write(str(riceAMLObj[feature].mean()))
    f.write(",")
    f.write(str(riceAMLObj[feature].median()))
    f.write(",")
    for mode in riceAMLObj[feature].mode():
        f.write(str(mode))
        f.write(" ")
    f.write(",")
    f.write(str(riceAMLObj[feature].std()))
    f.write("\n")

    print(feature, " Max:", riceAMLObj[feature].max())
    print(feature, " Min:", riceAMLObj[feature].min())
    print(feature, " Mean:", riceAMLObj[feature].mean())
    print(feature, " Median:", riceAMLObj[feature].median())
    print(feature, " Mode:", riceAMLObj[feature].mode())
    print(feature, " Std:", riceAMLObj[feature].std())
