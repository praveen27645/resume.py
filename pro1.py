def get_marks(count=6):
	print(f"Enter marks for {count} students.")
	print("You can enter all marks on one line separated by spaces, or enter them one per line.")
	marks = []
	line = input().strip()
	if line:
		for tok in line.split():
			try:
				marks.append(float(tok) if '.' in tok else int(tok))
			except ValueError:
				pass
	while len(marks) < count:
		try:
			val = input(f"Mark {len(marks)+1}: ").strip()
			marks.append(float(val) if '.' in val else int(val))
		except ValueError:
			print("Invalid input, please enter a number.")
	return marks[:count]


def main():
	marks = get_marks(6)
	asc = sorted(marks)
	desc = sorted(marks, reverse=True)
	print("\nSorted marks (ascending):")
	print(asc)
	print("\nSorted marks (descending):")
	print(desc)


if __name__ == '__main__':
	main()
