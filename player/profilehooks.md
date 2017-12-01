profilehooks输出每列的具体解释如下：
ncalls：表示函数调用的次数；
tottime：表示指定函数的总的运行时间，除掉函数中调用子函数的运行时间；
percall：（第一个percall）等于 tottime/ncalls；
cumtime：表示该函数及其所有子函数的调用运行的时间，即函数开始调用到返回的时间；
percall：（第二个percall）即函数运行一次的平均时间，等于 cumtime/ncalls；
filename:lineno(function)：每个函数调用的具体信息；

排序方式使用的是函数调用时间(cumulative)，除了这个还有一些其他允许的排序方式：calls, cumulative, file, line, module, name, nfl, pcalls, stdname, time等

*** PROFILER RESULTS ***
suggest_move_mcts (/Users/wangjian/WORK_ROOT/chess_deeplearning/player/MCTSPlayer_C.pyx:78)
function called 1 times

         1193801 function calls (1193558 primitive calls) in 3.439 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 338 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1898    0.016    0.000    3.372    0.002 tasks.py:224(_step)
     1597    0.004    0.000    3.361    0.002 tasks.py:308(_wakeup)
     1898    0.071    0.000    3.351    0.002 {method 'send' of 'coroutine' objects}
       81    0.001    0.000    1.458    0.018 policy.py:44(<lambda>)
       81    0.002    0.000    1.458    0.018 tensorflow_backend.py:2259(__call__)
       81    0.001    0.000    1.443    0.018 session.py:787(run)
       81    0.006    0.000    1.441    0.018 session.py:1040(_run)
       81    0.001    0.000    1.397    0.017 session.py:1262(_do_run)
       81    0.000    0.000    1.393    0.017 session.py:1325(_do_call)
       81    0.002    0.000    1.392    0.017 session.py:1294(_run_fn)
       81    1.387    0.017    1.387    0.017 {built-in method _pywrap_tensorflow_internal.TF_Run}
      301    0.050    0.000    0.751    0.002 Node.py:43(expand_node)
     7749    0.040    0.000    0.570    0.000 Node.py:226(play_move)
     3445    0.015    0.000    0.458    0.000 __init__.py:1639(log)
     3365    0.007    0.000    0.453    0.000 __init__.py:1598(debug)
     3445    0.013    0.000    0.415    0.000 __init__.py:1398(_log)
     7749    0.020    0.000    0.278    0.000 __init__.py:3177(copy)
     7749    0.024    0.000    0.258    0.000 __init__.py:1072(copy)
     3445    0.006    0.000    0.249    0.000 __init__.py:1423(handle)
     2434    0.009    0.000    0.247    0.000 Node.py:101(select_action_by_score)
     3445    0.016    0.000    0.240    0.000 __init__.py:1477(callHandlers)
     2434    0.024    0.000    0.238    0.000 {built-in method builtins.max}
     7749    0.145    0.000    0.233    0.000 __init__.py:1181(__init__)
     3445    0.009    0.000    0.224    0.000 __init__.py:848(handle)
    56397    0.029    0.000    0.214    0.000 Node.py:102(<lambda>)
     3445    0.012    0.000    0.206    0.000 __init__.py:974(emit)
     7749    0.077    0.000    0.199    0.000 __init__.py:1795(push)
    13518    0.028    0.000    0.193    0.000 __init__.py:3027(generate_legal_moves)
    56397    0.179    0.000    0.185    0.000 Node.py:91(get_value)
     3445    0.013    0.000    0.151    0.000 handlers.py:102(format)
      301    0.001    0.000    0.147    0.000 features.py:5(extract_features)
     2734    0.003    0.000    0.141    0.000 Node.py:71(is_game_over)
     2734    0.011    0.000    0.138    0.000 __init__.py:1574(is_game_over)
     3445    0.005    0.000    0.125    0.000 __init__.py:825(format)
     3445    0.008    0.000    0.120    0.000 formatter.py:139(format)
     3445    0.015    0.000    0.118    0.000 __init__.py:1383(makeRecord)
     4941    0.005    0.000    0.108    0.000 {built-in method builtins.any}
     3445    0.010    0.000    0.105    0.000 formatter.py:129(format)
     3445    0.055    0.000    0.103    0.000 __init__.py:249(__init__)
    13586    0.040    0.000    0.103    0.000 __init__.py:1249(generate_pseudo_legal_moves)