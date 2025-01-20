tokens = ['<CLS>','A','T','C','G']

tokenonthot = {token: i for i, token in enumerate(tokens)}
tokenonthot['N'] = 0
onhottoken = {i: token for i, token in enumerate(tokens)}