from TreeHMM4Glycan.Glycan import Glycan

import csv

if __name__ == "__main__":
    iupac_name_file = './Data/IUPAC.csv'
    with open(iupac_name_file) as file_in:
        csv_reader = csv.reader(file_in)
        for idx,row in enumerate(csv_reader):
            if idx == 0:
                continue
            case = row[1]
            count = int(row[2])
            if count > 0:
                glycan = Glycan(case)
                print(case)
                print(glycan.get_emssions_with_linkage())
                print(glycan.get_adj_list())
                #print(glycan.get_emssions())