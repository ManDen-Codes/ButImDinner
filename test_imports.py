import sys

def test(name, fn):
    try:
        fn()
        print(f"OK: {name}")
    except Exception as e:
        print(f"FAIL: {name} -> {e}")
        sys.exit(1)

test("torch", lambda: __import__("torch"))
test("yaml", lambda: __import__("yaml"))
test("datasets", lambda: __import__("datasets"))
test("peft", lambda: __import__("peft"))
test("transformers", lambda: __import__("transformers"))
test("trl", lambda: __import__("trl"))
test("trl.SFTTrainer", lambda: __import__("trl", fromlist=["SFTTrainer"]))
test("trl.SFTConfig", lambda: __import__("trl", fromlist=["SFTConfig"]))

print("All imports OK")
