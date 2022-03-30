import pandas as pd

#read csv
riceAMLObj = pd.read_csv('Rice_MSC_Dataset.csv', sep=',')
#print shape
print("Features and attributes of dataset: ", riceAMLObj.shape)
#features we are extracting from the dataset, excluding the image color features
featureList = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ","SOLIDITY","CONVEX_AREA", "EXTENT","ASPECT_RATIO","ROUNDNESS","COMPACTNESS","SHAPEFACTOR_1","SHAPEFACTOR_2","SHAPEFACTOR_3","SHAPEFACTOR_4"]
#creating rice stats csv
f = open("riceStats.csv", "a")

classList = ["Basmati", "Arborio", "Jasmine", "Ipsala", "Karacadag"]

for item in classList:

    riceAMLObjSubset = riceAMLObj.loc[(riceAMLObj["CLASS"] == item)]
    print(item)
    print(riceAMLObjSubset)
    f.write(item)
    f.write('\n')
    #writing to file
    f.write("name,max,min,mean,median,mode,standard deviation\n")
    for feature in featureList:
        f.write(feature)
        f.write(",")
        f.write(str(riceAMLObjSubset[feature].max()))
        f.write(",")
        f.write(str(riceAMLObjSubset[feature].min()))
        f.write(",")
        f.write(str(riceAMLObjSubset[feature].mean()))
        f.write(",")
        f.write(str(riceAMLObjSubset[feature].median()))
        f.write(",")
        for mode in riceAMLObjSubset[feature].mode():
            f.write(str(mode))
            f.write(" ")
        f.write(",")
        f.write(str(riceAMLObjSubset[feature].std()))
        f.write("\n")

        #print(feature, " Max:", riceAMLObjSubset[feature].max())
        #print(feature, " Min:", riceAMLObjSubset[feature].min())
        #print(feature, " Mean:", riceAMLObjSubset[feature].mean())
        #print(feature, " Median:", riceAMLObjSubset[feature].median())
        #print(feature, " Mode:", riceAMLObjSubset[feature].mode())
        #print(feature, " Std:", riceAMLObjSubset[feature].std())