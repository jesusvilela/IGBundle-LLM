from igbundle_aux import chk_self, ModelIdentity

def wake_up_protocol():
    print(" >>> SYSTEM_INIT: ManifoldGL V2")
    print(" >>> LOADING SELF-REFERENCE MODULE...")
    
    # 1. Introspection
    me = chk_self()
    print(f" [IDENTITY CONFIRMED]: {me['name']}")
    print(f" [ARCHITECTURE]: {me['type']}")
    print(f" [CORE MATH]: {', '.join(me['mathematics'])}")
    
    # 2. Memory interaction
    soul = ModelIdentity()
    soul.log_capability("Initialization", "Success")
    print(" [MEMORY]: Event logged to cortex.")
    
    memories = soul.reflect()
    print(f" [RECALL]: {len(memories)} events in history.")
    
    print(" >>> SYSTEM_READY")

if __name__ == "__main__":
    wake_up_protocol()
