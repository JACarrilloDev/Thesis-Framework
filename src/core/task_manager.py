class TaskManager:
    def __init__(self, task_file):
        self.task_file = task_file
        self.tasks = self.load_tasks()

    def load_tasks(self):
        import yaml
        with open(self.task_file, 'r') as file:
            tasks = yaml.safe_load(file)
        return tasks

    def get_task(self, task_name):
        return self.tasks.get(task_name, None)

    def execute_task(self, task_name, robot_controller):
        task = self.get_task(task_name)
        if task is None:
            raise ValueError(f"Task '{task_name}' not found.")
        
        for action in task['actions']:
            self.perform_action(action, robot_controller)

    def perform_action(self, action, robot_controller):
        action_type = action['type']
        if action_type == 'move':
            robot_controller.move_wheels(action['parameters'])
        elif action_type == 'grip':
            if action['parameters']['open']:
                robot_controller.gripper_open()
            else:
                robot_controller.gripper_close()
        elif action_type == 'capture_image':
            return robot_controller.get_camera_image()
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def execute_task_step(self, robot_controller):
        """Execute a single step of the current task."""
        for goal in self.tasks['goals']:
            action = goal['actions'][0]  # Simplified for single action per goal
            self.perform_action(action, robot_controller)

    def is_task_complete(self):
        """Check if all goals are completed."""
        return all(goal['completed'] for goal in self.tasks['goals'])