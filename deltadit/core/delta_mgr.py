from deltadit.utils.logging import logger

DELTA_MANAGER = None


class DELTAConfig:
    def __init__(
        self,
        steps: int,
        delta_skip: bool,
        delta_threshold: list,
        delta_gap: int,
    ):
        self.steps = steps

        self.delta_skip = delta_skip
        self.delta_threshold = delta_threshold  # {(0,10):[0:9], (0,30):[9:28]}
        self.delta_gap = delta_gap

        # delta_threshold = {(0,10):[0,9], (10,30):[9,28]}
        self.current_interval, self.time_intervals, self.start_block_id, self.end_block_id = None, None, None, None

    def setup(self, t):
        (
            self.current_interval,
            self.time_intervals,
            self.start_block_id,
            self.end_block_id,
        ) = self.extract_keys_and_assign_blocks(self.delta_threshold, t)

    def extract_keys_and_assign_blocks(self, time_dict, t):
        time_intervals = [range(interval[0], interval[1] + 1) for interval in time_dict.keys()]

        for interval in time_intervals:
            start_time, end_time = interval[0], interval[-1]
            if start_time <= t <= end_time:
                start_block_id = time_dict[(interval[0], interval[-1])][0]
                end_block_id = time_dict[(interval[0], interval[-1])][-1]
                return interval, time_intervals, start_block_id, end_block_id

        return None, time_intervals, None, None


class DELTAManager:
    def __init__(self, config: DELTAConfig):
        self.config: DELTAConfig = config
        self.cache = {self.config.start_block_id: None, self.config.end_block_id: None}
        self.count = 0

        init_prompt = f"Init DELTAManager. steps: {config.steps}."
        init_prompt += f" delta_skip: {config.delta_skip}, delta_threshold: {config.delta_threshold}, delta_gap: {config.delta_gap}."
        logger.info(init_prompt)

    def if_skip_delta(self, timestep: int):
        self.config.setup(timestep)
        # NOTE
        if self.config.current_interval is None:
            print("No skip interval")
            self.count = 1
        else:
            self.count = timestep - self.config.current_interval[0] + 1

        if (
            self.config.delta_skip
            and (timestep is not None)
            and (self.count % self.config.delta_gap == 0)
            and (self.is_t_in_intervals(timestep))
        ):
            flag = True
        else:
            flag = False
        # print(f"timestep: {timestep}, count: {self.count}")
        # self.count = (self.count + 1) % self.config.steps
        return flag

    def if_skip_middle_block(self, t, block_id):
        if (t is not None) and (self.count % self.config.delta_gap == 0) and (self.is_t_in_intervals(t)):
            if self.config.start_block_id <= block_id < self.config.end_block_id:
                flag = True
            else:
                flag = False
        else:
            flag = False
        return flag

    def is_t_in_intervals(self, t):
        return any(t in interval for interval in self.config.time_intervals)

    def save_start_cache(self, block_id, cache):
        self.cache[block_id] = cache

    def get_start_cache(self, block_id):
        return self.cache[block_id]

    def save_end_cache(self, block_id, cache):
        self.cache[block_id] = cache

    def get_end_cache(self, block_id):
        return self.cache[block_id]

    def get_cache(self):
        start = self.cache[self.config.start_block_id]
        end = self.cache[self.config.end_block_id]
        cache = start - end
        return cache

    def is_skip_last_block(self, block_id):
        if self.config.end_block_id is not None and block_id == self.config.end_block_id:
            return True
        else:
            return False

    def is_skip_first_block(self, block_id):
        if self.config.start_block_id is not None and block_id == self.config.start_block_id:
            return True


def set_delta_manager(config: DELTAConfig):
    global DELTA_MANAGER
    DELTA_MANAGER = DELTAManager(config)


def enable_delta():
    if DELTA_MANAGER is None:
        return False
    return DELTA_MANAGER.config.delta_skip


def update_steps(steps: int):
    if DELTA_MANAGER is not None:
        DELTA_MANAGER.config.steps = steps


def get_count():
    if DELTA_MANAGER is not None:
        return DELTA_MANAGER.count
    return None


def if_skip_delta(timestep: int):
    if not enable_delta():
        return False
    return DELTA_MANAGER.if_skip_delta(timestep)


def if_skip_middle_block(t, block_id):
    if not enable_delta():
        return False
    return DELTA_MANAGER.if_skip_middle_block(t, block_id)


def is_skip_last_block(block_id):
    if not enable_delta():
        return False
    return DELTA_MANAGER.is_skip_last_block(block_id)


def is_skip_first_block(block_id):
    if not enable_delta():
        return False
    return DELTA_MANAGER.is_skip_first_block(block_id)


def save_start_cache(block_id, cache):
    if not enable_delta():
        return
    DELTA_MANAGER.save_start_cache(block_id, cache)


def get_start_cache(block_id):
    if not enable_delta():
        return None
    return DELTA_MANAGER.get_start_cache(block_id)


def save_end_cache(block_id, cache):
    if not enable_delta():
        return
    DELTA_MANAGER.save_end_cache(block_id, cache)


def get_end_cache(block_id):
    if not enable_delta():
        return None
    return DELTA_MANAGER.get_end_cache(block_id)


def is_t_in_intervals(t):
    if not enable_delta():
        return False
    return DELTA_MANAGER.is_t_in_intervals(t)


def get_cache():
    if not enable_delta():
        return None
    return DELTA_MANAGER.get_cache()
