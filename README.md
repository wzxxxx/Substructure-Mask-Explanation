# Structure Mask Explanation (SME)
 structure-based explanation methods

Structure-Mask Explanation (SME) is an intuitive and simple perturbation-based explanation methods that identifies the substructure affecting predictions. Three different molecular fragmentation methods, BRICS, Murcko Scaffold, and functional group, were used in SME to gain a more comprehensive understanding of the relationship between substructure and properties. In addition, by analyzing the contribution of functional groups across the entire dataset, we were able to understand how functional groups affected model predictions, which in turn provided guidance for structure optimization. Both the prediction results of the model and the real-world structure optimization results confirm that the contribution of functional groups can reasonably guide the structure optimization. Moreover, the recombination of the BRICS fragments with attribution assigned by SME can be used to generate molecules with desired properties, which provides a new way to generate molecules with desired properties that does not require training.

![image](<https://github.com/wzxxxx/Structure-Mask-Explanation--SME-/tree/main/figure/SME.png>)



**requirements：**
python == 3.7
anaconda
dgl==0.7.1
rdkit==2020.09.1.0
pytorch=1.11.0
scikit-learn == 1.0.2
seaborn==0.11.2
numpy
pandas
scipy
pandas
numpynumpy
pandas
numpy
