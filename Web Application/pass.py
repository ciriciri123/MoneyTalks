import bcrypt
print(bcrypt.hashpw(b'admin123', bcrypt.gensalt(rounds=12)).decode())
