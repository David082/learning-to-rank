# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/7
version :
refer :
https://github.com/dmlc/xgboost/issues/3291
"""
import treelite

builder = treelite.ModelBuilder(num_feature=3)

###### First tree ######
tree = treelite.ModelBuilder.Tree()
# Node #0: feature 0 < 5.0 ? (default direction left)
tree[0].set_numerical_test_node(feature_id=0,
                                opname='<',
                                threshold=5.0,
                                default_left=True,
                                left_child_key=1,
                                right_child_key=2)
# Node #2: leaf with output +0.6
tree[2].set_leaf_node(0.6)
# Node #1: feature 2 < -3.0 ? (default direction right)
tree[1].set_numerical_test_node(feature_id=2,
                                opname='<',
                                threshold=-3.0,
                                default_left=False,
                                left_child_key=3,
                                right_child_key=4)
# Node #3: leaf with output -0.4
tree[3].set_leaf_node(-0.4)
# Node #4: leaf with output +1.2
tree[4].set_leaf_node(1.2)
# Set node #0 as root
tree[0].set_root()
# Insert the first tree into the ensemble
builder.append(tree)

###### Second tree ######
tree2 = treelite.ModelBuilder.Tree()
# Node #0: feature 1 < 2.5 ? (default direction right)
tree2[0].set_numerical_test_node(feature_id=1,
                                 opname='<',
                                 threshold=2.5,
                                 default_left=False,
                                 left_child_key=1,
                                 right_child_key=2)
# Set node #0 as root
tree2[0].set_root()
# Node #1: leaf with output +1.6
tree2[1].set_leaf_node(1.6)
# Node #2: feature 2 < -1.2 ? (default direction left)
tree2[2].set_numerical_test_node(feature_id=2,
                                 opname='<',
                                 threshold=-1.2,
                                 default_left=True,
                                 left_child_key=3,
                                 right_child_key=4)
# Node #3: leaf with output +0.1
tree2[3].set_leaf_node(0.1)
# Node #4: leaf with output -0.3
tree2[4].set_leaf_node(-0.3)

# Insert the second tree into the ensemble
builder.append(tree2)

## Finalize and obtain Model object
model = builder.commit()

# Export the model as XGBoost format by writing
model.export_as_xgboost('test.model', name_obj='binary:logistic')
