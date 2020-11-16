from TreeHMM4Glycan.Glycan import Glycan

case1 = '(3S)Gal(b1-4)[Fuc(a1-3)](6S)Glc'
case2 = '6S(3S)Gal(b1-4)[Fuc(a1-3)][Fuc(a1-3)]Glc'
case3 =  'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)]Man(b1-4)GlcNAc(b1-4)GlcNAc'
case4 = 'KDN(a2-3)Gal(b1-3)GlcNAc(a1-Sp0'
#'GlcNAc(b1-6)[GlcNAc(b1-2)](3S)Man(a1-6)[GlcNAc(b1-4)][GlcNAc(b1-4)[GlcNAc(b1-2)]Man(a1-3)]Man'
#glycan_from_iupac(case3)
#case2_glycan = Glycan(case2)
#print(case2_glycan.get_emssions())
for case in [case1, case2, case3, case4]:
    glycan = Glycan(case)
    print(case)
    print(glycan.get_emssions_with_linkage())
    print(glycan.get_adj_list())
    print(glycan.get_emssions())
    print(glycan.get_adj_matrix())