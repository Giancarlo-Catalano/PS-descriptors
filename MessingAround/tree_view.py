from anytree import Node, RenderTree

# Define your tree:
root = Node("Root")
left = Node("Left", parent=root)
right = Node("Right", parent=root)
left_child = Node("Left Child", parent=left)

# Print an ASCII representation of the tree:
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")
