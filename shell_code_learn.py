def turn_on_lamp():
    print("Lampa vkluchena!")


len_=10
push_i = 0
push_fl = 1
push_str = 2
r = 3

ops=["r","push_i","push_str","push_fl"]
def console():
    global b_c
    input_ = ""
    # splitted_cmd и splitted_cmd_src - т.к. работаем со статическим массивом
    splitted_cmd: list = [''] * 2
    splitted_cmd_src:list = None
    main_cmd = '<uninitialized>'
    par_cmd = '<uninitialized>'
    idex_of_bytecode_is_bytecode = 0
    cmd_in_ops = '<uninitialized>'
    pos_bytecode = -1
    shell_is_running = True
    print("Zdravstvuite ya sostavitel bait-coda dla etoi programmi")
    print("Dostupnie codi")
    for c in ops:
        print(c, end=' ')
    print()
    while shell_is_running:
        
        input_ = input(">>>")
        if input_== "r":
            break
        
        splitted_cmd_src = input_.split()
        for pos_to_write in range(len(splitted_cmd_src)):
            splitted_cmd[pos_to_write] = splitted_cmd_src[pos_to_write]
        main_cmd = splitted_cmd[0]
        par_cmd = splitted_cmd[1]
        # Ищем код в списке код-строку
        for idex_of_bytecode_is_bytecode in range(len(ops)):
            cmd_in_ops = ops[idex_of_bytecode_is_bytecode]
            if cmd_in_ops == main_cmd:
                pos_bytecode += 1
                # формируем числовой байт-код и если нужно значения параметра
                b_c[pos_bytecode] = idex_of_bytecode_is_bytecode
                if par_cmd != '':
                    pos_bytecode += 1
                    b_c[pos_bytecode] = par_cmd
            # Очищаем
            splitted_cmd[0] = ''
            splitted_cmd[1] = ''
        pos_bytecode= + 1

def vm_proc_to_learn(b_c:list):
    ip=0
    sp=-1
    sp_str=-1
    sp_fl=-1
    steck=[0]*len_
    steck_fl=[0.0]*len_
    steck_str=['']*len_
    op=0
    op=b_c[ip]
    while True:
        if op==push_i:
            sp+=1
            ip+=1
            steck[sp]=int(b_c[ip]) # Из строкового параметра
        elif op == push_fl:
            sp_fl += 1
            ip += 1
            steck_fl[sp_fl] = float(b_c[ip])  # Из строкового параметра
        elif op==push_str:
            sp_str+= 1
            ip += 1
            steck_str[sp_str] = b_c[ip]
        elif op == r:
            break
       
