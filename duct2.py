dict = {}
dict['key'] = 'value'
print(dict['key'])
marks = {"harry": 85, "ron": 78, "hermione": 92, "draco": 67, "luna": 88}
print(marks, type(marks))
print("Keys:", marks.keys())
print("Values:", marks.values())
print(marks.items())
print(marks.get("harry"))
marks.update({"neville": 73}) #prints none
print(marks)