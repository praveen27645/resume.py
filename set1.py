s = {1, 5, 32, 12, 7, 3}
print("Original set:", s)
print("Type of s:", type(s))                                
s.add(8)
print("Set after adding 8:", s)
s.remove(3)
print(s, type(s))
s.clear()
print("Set after clearing:", s)
s1 = {1, 2, 3, 4, 5}
s2 = {4, 5, 6, 7, 8}
print(s1.union(s2))
print(s1.intersection(s2))
print("Set 1:", s1)
print("Set 2:", s2)
print("Intersection of s1 and s2:", s1.intersection(s2))
print("Difference of s1 and s2 (s1 - s2):", s1.difference(s2))
print("Symmetric difference of s1 and s2:", s1.symmetric_difference(s2))
print(s1.union(s2))