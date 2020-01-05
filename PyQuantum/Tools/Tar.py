import tarfile

def to_targz(data, filename):
	f = tarfile.open(name=filename, mode='w:gz', fileobj=None, bufsize=10240)
	
	for i in data:
		print(i)
		f.add(i)
	
	f.close()

