import os,sys
import re
import json
import argparse
import queue
import time
import signal 
import subprocess
import concurrent.futures
from queue import Queue
from loguru import logger
import multiprocessing
import datetime
from multiprocessing import Manager

run_log_dir = None
run_name = None

def print_fmt(message, level="INFO", flag="CI_LOG"):
    global run_log_dir
    timestamp = datetime.datetime.now().strftime("%d %H:%M:%S")
    prefix = f"[{timestamp}][{level}][{flag}]"
    full_message = f"{prefix} {message}"
    print(full_message)
    log_file = os.path.join(run_log_dir, "ci_result_summary.log")
    with open(log_file, 'a', encoding='utf-8') as file:
        file.write(f"{full_message}\n")

def set_log_dir(dir):
    global run_log_dir
    run_log_dir = dir
    os.makedirs(run_log_dir, exist_ok=True)

def set_log_dir_by_run_name(l_run_name):
    global run_name
    run_name = l_run_name
    CASE_WORK_DIR = os.environ.get("TRITON_WORKSPACE")
    base_log_dir=os.path.join(CASE_WORK_DIR, "ci_log")
    run_log_dir = os.path.join(base_log_dir, run_name)
    set_log_dir(run_log_dir)

def set_log_dir_by_op(ops_name):
    global run_log_dir
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    l_run_name = f"{ops_name}_{timestamp}"
    set_log_dir_by_run_name(l_run_name)

##test_set_name必须在json文件中有定义
def read_json_ops_and_tasks(file_path, test_set_name=None):
    all_op_list = []
    test_op_list = None
    all_task_dict = {}
    hardware_bug_ops = []
    time_consume_ops = []
    print_fmt(f"{file_path}", "info", "Triton CI Run Flaggems OPs")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_op_list = json_data.get('all_ops', [])
            if test_set_name is not None:
                test_op_list = json_data.get(test_set_name, [])
            all_task_dict = json_data.get('all_tasks', {})
            hardware_bug_ops = json_data.get('hardware_bug_ops', [])
            time_consume_ops = json_data.get('time_consume_ops', [])
    except FileNotFoundError:
        print_fmt(f" {file_path} not found!", "error", "Triton CI Run Flaggems OPs")
    except json.JSONDecodeError:
        print_fmt(f" {file_path} data decode error!", "error", "Triton CI Run Flaggems OPs")
    except Exception as e:
        print_fmt(f" {file_path} read json fail!", "error", "Triton CI Run Flaggems OPs")

    return all_op_list, test_op_list, all_task_dict, hardware_bug_ops, time_consume_ops


def check_card_status(card_id: str):
    """
    执行tsm_smi 解析NPU-Util的值
    Args:
        card_id (str): 输入card id or 返回所有卡的信息
    Returns:
        bool: True表示空闲(0%), False表示被占用
    """
    try:
        output = subprocess.check_output(
            "tsm_smi",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"命令执行异常: {e}")
        return False

    # logger.debug(f"output: {output}")
    lines = output.splitlines()

    for line in lines:
        # logger.debug(f"line: {line}")
        line = re.sub(r'\s{2,}', ' ', line.strip())
        line = line.replace("'", "").replace("/", "").replace("|", "").strip()
        if not line.strip() or "Card count" in line or "TSM-SMI" in line or "NPU-Util" in line:
            continue
        if "Card" in line and "Name" in line and "NPU-Util" in line:
            continue

        fields = line.split()
        # logger.debug(f"fields: {fields}")
        if len(fields) < 2:
            continue

        if str(fields[0]) != card_id and str(fields[1]) != card_id and str(fields[2]) != card_id:
            continue

        for field in fields :
            if field.endswith("%"):
                util_str = field.strip("%")
                if util_str.isdigit():
                    util_value = int(util_str)
                    # logger.debug(f'util_value: {util_value}')
                    return util_value == 0

    logger.error(f"ERROR: 未找到卡号 {card_id} ")
    return False

def check_card_status_i(card_id: str):
    try:
        output = subprocess.check_output(
            f"tsm_smi -i {card_id}",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"命令执行异常: {e}")
        return False

    lines = output.splitlines()

    for line in lines:
        if "No running processes found" in line:
            return True

    logger.warning(f"warning: 卡被占用: {card_id}")
    return False

def run_case(task, all_tasks, card_id, quick_mode, time_consume_ops, l_run_name):
    """
    运行各算子的test_accuracy函数,得到算子验证结果
    Args:
        task: 算子测试函数
        card_id: 当前使用的npu id
    """
    set_log_dir_by_run_name(l_run_name)

    triton_cache_dir = f"/tmp/triton_log/{l_run_name}/triton_cache_" + card_id 
    triton_dump_dir = f"/tmp/triton_log/{l_run_name}/triton_dump_" + card_id 
    flaggems_cache_dir = f"/tmp/triton_log/{l_run_name}/flaggems_cache_" + card_id 
    txda_skip_ops = os.getenv("TXDA_SKIP_OPS", "")
    if task == 'to_dtype':
        txda_skip_ops += ',to.dtype'
    env = os.environ.copy()
    env.update({
        'TXDA_VISIBLE_DEVICES': card_id,
        #'PRECISION_PRIORITY': "1",
        'TRITON_CACHE_DIR': triton_cache_dir,
        #'TRITON_DUMP_PATH': triton_dump_dir,
        'FLAGGEMS_CACHE_DIR': flaggems_cache_dir,
        'TXDA_SKIP_OPS': txda_skip_ops,
    })
    # print_fmt(f"current case: {task}\ncard_id: {card_id}\nenviroment: {env}")
    
    #case_dir = os.path.join(CASE_WORK_DIR, task)
    log_dir = os.path.join(run_log_dir, "flag_gems")
    os.makedirs(log_dir, exist_ok=True)

    interval = 60
    threshold = 5
    if task in time_consume_ops:
        quick_mode = 1
        threshold = 50
    op_funcs = all_tasks[task]
    file_path = os.path.dirname(os.path.abspath(__file__))
    failed_op_func_count = 0
    succ_op_func_count = 0
    failed_op_func_list = []
    print_fmt(f"{task} begin test...", "info", "Triton CI Run Flaggems OPs")
    for t_func in op_funcs:
        log_file_path = os.path.join(log_dir, f"{t_func}.log")
        print_fmt(f"{t_func} start test >>>>>>", "info", "Triton CI Run Flaggems OPs")
        print_fmt(f"log_file_path: {log_file_path}", "info", "Triton CI Run Flaggems OPs")
        cmd = ["python3", "-m", "pytest", "-v", "-s", f"{file_path}/{t_func}", "--ref", "cpu"]
        if quick_mode:
            cmd.append("--mode")
            cmd.append("quick")
        print_fmt(f"cmd: {' '.join(cmd)}", "info", "Triton CI Run Flaggems OPs")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(file=log_file_path, mode='w'),
            stderr=subprocess.STDOUT
        )
        
        counter = 0
        prev_size = 0      
        while True:
            time.sleep(interval)  
            try:
                current_size = os.path.getsize(log_file_path)
            except FileNotFoundError:
                print_fmt(f" log file {log_file_path} not exists!", "error", "Triton CI Run Flaggems OPs")
                break
            
            if current_size == prev_size:
                counter += 1
                print_fmt(f"{t_func} log file size unchanged，already：{counter}/{threshold} times!", "info", "Triton CI Run Flaggems OPs")
                if counter >= threshold:
                    print_fmt(f" {t_func} already stuck，timeout and stoped!", "error", "Triton CI Run Flaggems OPs")
                    # process.terminate()
                    # 发送ctrl+c信号进行资源释放
                    process.send_signal(signal.SIGINT)   
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break  
            else:
                counter = 0
                prev_size = current_size
                print_fmt(f"{t_func} log is normal，current_size：{current_size} Byte.", "info", "Triton CI Run Flaggems OPs")
            
            # 判断该进程是否还在进行
            if process.poll() is not None:
                break
            
            time.sleep(10)      
        # 等待进程结束
        process.wait()
        time.sleep(10)
        if process.returncode == 0:
            print_fmt(f"{t_func} test success!", "info", "Triton CI Run Flaggems OPs")
            succ_op_func_count += 1
        else:
            print_fmt(f"{t_func} test failed, ret_code: {process.returncode}", "error", "Triton CI Run Flaggems OPs")
            failed_op_func_count += 1
            failed_op_func_list.append((t_func, process.returncode))
    if failed_op_func_count > 0:
        print_fmt(f"{task} completed, {failed_op_func_count}/{failed_op_func_count+succ_op_func_count} func failed.", "error", "Triton CI Run Flaggems OPs")
        return failed_op_func_count, task, failed_op_func_list
    else:
        print_fmt(f"{task} completed, all func success.", "info", "Triton CI Run Flaggems OPs")
        return 0, task, []

def run_stage(args):
    global run_name
    test_set_name = args.test_set
    quick_mode = args.quick
    json_file_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = json_file_dir + "/flag_gems_ci_ops.json"
    all_op_list, test_op_list, all_tasks, hardware_bug_ops, time_consume_ops = read_json_ops_and_tasks(json_file_path, test_set_name)
    NPU_IDS = [str(i) for i in range(args.device_count) if i not in args.skip_device]
    if test_op_list is None:
        test_op_list = all_op_list
        print_fmt(f"{test_set_name} is None, use all_op_list", "[warn]", "Triton CI Run Flaggems OPs")

    multiprocessing.set_start_method("spawn")
    with Manager() as manager:
        task_queue = manager.Queue()
        card_queue = manager.Queue()
        pass_queue = manager.Queue()
        fail_queue = manager.Queue()

        for op in test_op_list:
            if op in all_tasks:
                task_queue.put(op)
            else:
                print_fmt(f"{op} not in all_tasks, discard!", "[warn]", "Triton CI Run Flaggems OPs")

        for card_id in NPU_IDS:
            if check_card_status_i(card_id):
                card_queue.put(card_id)
            else:
                print_fmt(f"Card {card_id} is not available, discard!", "[warn]", "Triton CI Run Flaggems OPs")
        
        process_count = min(card_queue.qsize(), task_queue.qsize())
        total_count = task_queue.qsize()
        print_fmt(f"[Triton CI Run Flaggems OPs][info]total task count:{total_count}")
        print_fmt(f"[Triton CI Run Flaggems OPs][info]process count:{process_count}")
        # 修改回调函数，当出现case异常时，卡则不再放回资源池中
        def callback(future, card_id):
            try:
                result, op_name, failed_op_func_list = future.result()
                if result == 0:
                    pass_queue.put(op_name)
                else:
                    fail_queue.put((op_name, failed_op_func_list))
                if (check_card_status_i(card_id)):
                    card_queue.put(card_id)
                    print_fmt(f"Card {card_id} released and available!", "info", "Triton CI Run Flaggems OPs")
                else:
                    print_fmt(f"Card {card_id} is not available, not recycled!", "[warn]", "Triton CI Run Flaggems OPs")
            except Exception as e:
                print_fmt(f"Task Exception: {e}, Card {card_id} is not available, not recycled!", "error", "Triton CI Run Flaggems OPs")
        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as excutor:
            futures = {}
            for _ in range(process_count):
                case_dir = task_queue.get()
                card_id = card_queue.get()
                future = excutor.submit(run_case, case_dir, all_tasks, card_id, quick_mode, time_consume_ops, run_name)
                future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                futures[future] = (case_dir, card_id)
                print_fmt(f"Started: {total_count-task_queue.qsize()}: {case_dir} on card {card_id}", "info", "Triton CI Run Flaggems OPs")

            while not task_queue.empty():
                completed, _ = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                # 处理已经完成的任务
                for future in completed:
                    case_dir, card_id = futures.pop(future)
                    print_fmt(f"Completed: {case_dir} on card {card_id}", "info", "Triton CI Run Flaggems OPs")

                # 根据剩余卡和任务数量提交新的任务
                available = card_queue.qsize()
                to_submit = min(available, task_queue.qsize())
                for _ in range(to_submit):
                    try:
                        case_dir = task_queue.get_nowait()
                        card_id = card_queue.get()
                        new_future = excutor.submit(run_case, case_dir, all_tasks, card_id, quick_mode, time_consume_ops, run_name)
                        new_future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                        futures[new_future] = (case_dir, card_id)
                        print_fmt(f"Started: {total_count-task_queue.qsize()}: {case_dir} on card {card_id}", "info", "Triton CI Run Flaggems OPs")
                    except queue.Empty:
                        break
            #处理最后完成的任务
            completed, _ = concurrent.futures.wait(
                list(futures.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                case_dir, card_id = futures.pop(future)
                print_fmt(f"Completed: {case_dir} on card {card_id}", "info", "Triton CI Run Flaggems OPs")
        print_fmt(f"Total ops number is: {total_count}") 
        succed_count = pass_queue.qsize()
        print_fmt(f"Passed ops number is: {succed_count}")
        for i in range(succed_count):
            print_fmt(f"\t{pass_queue.get()}")
        
        failed_and_not_in_decluded_ops = 0
        failed_count = fail_queue.qsize()
        print_fmt(f"Failed ops number is: {failed_count}")
        for i in range(failed_count):
            op_name, failed_op_func_list = fail_queue.get()
            print_fmt(f"{op_name}:")
            for (tfunc, retcode) in failed_op_func_list:
                print_fmt(f"\t{tfunc}: {retcode}")
            if op_name not in hardware_bug_ops:
                failed_and_not_in_decluded_ops += 1
        print_fmt("All tasks processed")
        if failed_and_not_in_decluded_ops/total_count < 0.05:
            return 0
        else:  
            return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlagGems OP accuracy test for Triton CI.")
    parser.add_argument("--test_set",
                        type = str,
                        help = "op set name to test, define in flag_gems_ci_ops.json, e.g. --test_set all_ops",
                        default ="all_ops")
    parser.add_argument('--quick', action='store_true', default=False,
                        help='run tests on quick mode')
    parser.add_argument("--device_count",
                        type = int,
                        help = "Maximum number of devices that can be used.",
                        default= 1)
    parser.add_argument("--skip_device",
                        type = int,
                        nargs='*',
                        help = "Devices that need to be skipped, when they are unavailable.",
                        default= [])
    args = parser.parse_args()

    set_log_dir_by_op(args.test_set)

    print_fmt("------------------all env---------------------", "info", "Triton CI Run Flaggems OPs")
    for key, value in os.environ.items():
        print_fmt(f"{key}={value}")
    print_fmt("----------------------------------------------", "info", "Triton CI Run Flaggems OPs")
    start_time = time.time()
    exit_code = run_stage(args)
    end_time = time.time()
    print_fmt(f"time cost: {(end_time - start_time):.2f}s", "info", "Triton CI Run Flaggems OPs")
    if exit_code is not None:
        if exit_code ==0:
            sys.exit(0)
        else:
            sys.exit(-1)
    else:
        sys.exit(-1)
