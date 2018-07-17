from ctypes import *
import ctypes, win32process
import psutil


class MemoryManager:
    """
    Memory manager contains methods for reading values from the memory of the application
    so we can use them as part of the reward function for reinforcement learning
    """
    def __init__(self):
        self.PID = self.get_client_pid("One Finger Death Punch.exe")
        # process = windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        # OpenProcess = windll.kernel32.OpenProcess
        self.ReadProcessMemory = windll.kernel32.ReadProcessMemory
        # FindWindowA = windll.user32.FindWindowA
        # GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId

        self.PROCESS_ALL_ACCESS = 0x1F0FFF
        #self.HWND = win32ui.FindWindow(None, u"One Finger Death punch").GetSafeHwnd()
        #self.PID = win32process.GetWindowThreadProcessId(HWND)[1]
        self.processHandle = ctypes.windll.kernel32.OpenProcess(self.PROCESS_ALL_ACCESS, False, self.PID)

        #print(f"HWND: {self.HWND}")
        #print(f"PID: {self.PID}")
        print(f"PROCESS: {self.processHandle}")

        # Open process for reading & writing:
        self.BaseAddress = win32process.EnumProcessModules(self.processHandle)[0]
        #print("Base memory address", hex(self.BaseAddress))

        # Read out the app base (this is not the program / module base, the 'app' is just a huge object):
        self.appBase = c_int()
        self.numRead = c_int()

        # NOT USED CURRENTLY
        game_memory_address = 0x00330000
        self.game = c_int()
        self.ReadProcessMemory(self.processHandle, self.BaseAddress + game_memory_address, byref(self.game), 4, byref(self.numRead))
        #print("Game memory address", hex(self.game.value))

    def read_kill_count(self):
        """
        Read the kill count variabel from memory for use as a reward
        :return: Kill count as int:
        """
        # Pick the memory address that is be overwriten automatically. This is the most stable address
        kill_count_memory_address = 0x006FB5E0
        kill_count = c_int()
        self.ReadProcessMemory(self.processHandle, kill_count_memory_address, byref(kill_count), 4,
                               byref(self.numRead))  # This works somewhat direct address 0x12FB840

        return kill_count.value

    def read_health(self):
        """
        Read the health variabel from memory for use as a negative reward when lost
        :return: Kill health as int between 0 and 10:
        """
        health_memory_address = 0x048A1198
        health = c_int()
        self.ReadProcessMemory(self.processHandle, health_memory_address, byref(health), 4,
                               byref(self.numRead))  # This works somewhat direct address 0x12FB840

        return health.value

    @staticmethod
    def get_client_pid(process_name):
        pid = None
        for process in psutil.process_iter():
            if process.name() == process_name:
                pid = int(process.pid)
                print("Found, PID = ", pid)
                break
        return pid