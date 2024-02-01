# %%
import xml.etree.ElementTree as ET
import tempfile

# Parse the XML file
tree = ET.parse('./tasks/quadruped/a1 copy.xml')

# Get the root of the XML document
root = tree.getroot()

# Initialize an empty set to store unique classes
unique_classes = set()

# Traverse the XML tree to find all unique classes
for elem in ryoot.iter():
    if 'class' in elem.attrib:
        unique_classes.add(elem.attrib['class'])

unique_classes
#%%
# Assume you have a dictionary mapping old class names to new ones
x = 0
class_mapping = {old_class: f"{old_class}_{x}" for old_class in unique_classes}

class_mapping

# %%
# Traverse the XML tree to find the elements with class attributes
for elem in root.iter():
    if 'class' in elem.attrib:
        old_class = elem.attrib['class']
        if old_class in class_mapping:
            elem.attrib['class'] = class_mapping[old_class]  # Change the class attribute

# Write the changes back to the XML file
tree.write('your_file.xml')

# %%
print(ET.tostring(root))
# %%
