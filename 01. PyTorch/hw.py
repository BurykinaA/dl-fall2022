import torch
import yaml

from abc import ABC
from typing import List

<<<<<<< HEAD
from torch import Tensor

=======
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0

class Task(ABC):
    def solve():
        """
        Function to implement your solution, write here
        """

    def evaluate():
        """
        Function to evaluate your solution
        """


class Task1(Task):
    """
        Calculate, using PyTorch, the sum of the elements of the range from 0 to 10000.
    """
<<<<<<< HEAD

=======
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0
    def __init__(self) -> None:
        self.task_name = "task1"

    def solve(self):
        # write your solution here
<<<<<<< HEAD
        return torch.tensor(range(10000)).sum()
=======
        pass
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.item()}}


class Task2(Task):
    """
        Solve optimization problem: find the minimum of the function f(x) = ||Ax^2 + Bx + C||^2, where
        - x is vector of size 8
        - A is identity matrix of size 8x8
        - B is matrix of size 8x8, where each element is 0
        - C is vector of size 8, where each element is -1

        Use PyTorch and autograd function. Relative error will be less than 1e-3
        
        Solution here is x, converted to the list(see submission.yaml).
    """
<<<<<<< HEAD

=======
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0
    def __init__(self) -> None:
        self.task_name = "task2"

    def solve(self):
        # write your solution here
<<<<<<< HEAD
        x = torch.rand(8, 1, requires_grad=True)
        A = torch.eye(8)
        B = torch.zeros(8, 8)
        C = torch.Tensor([-1] * 8)

        lr = 1e-2

        for _ in range(50):
            x.grad = None
            v = (A @ (x * x) + B @ x + C)
            loss = torch.linalg.norm(v) ** 2
            loss.backward()

            with torch.inference_mode():
                x -= x.grad * lr

            #print(loss)

        print(x.view(-1))

        return x.view(-1)
=======
        pass
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


<<<<<<< HEAD
class Linear(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()

        self.layer = torch.nn.Linear(in_shape, out_shape)

    def forward(self, x):
        return torch.transpose(self.layer(x), 0, 1)


=======
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0
class Task3(Task):
    """
        Solve optimization problem: find the optimal parameters of the linear regression model, using PyTorch.
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]

        text_X = [[0, -1], [-1, 0]]

        User PyTorch. Relative error will be less than 1e-1
        
        Solution here is test_y, calculated from test_X, converted to the list(see submission.yaml).
    """
<<<<<<< HEAD

=======
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0
    def __init__(self) -> None:
        self.task_name = "task3"

    def solve(self):
<<<<<<< HEAD
        lr = 1e-2

        train_X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float)
        train_y = torch.tensor([1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746])

        text_X = torch.tensor([[0, -1], [-1, 0]], dtype=torch.float)

        model = Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr)
        criterion = torch.nn.MSELoss()

        for idx in range(10000):
            optimizer.zero_grad()
            y_hat = model(train_X)
            y_hat = y_hat.view(-1)
            # print(y_hat)
            L = criterion(y_hat, train_y)
            # print(train_y)
            L.backward()
            optimizer.step()
            # if idx % 1000 == 0:
            #     print(f"Grad_norm: {list(model.parameters())[0].grad.norm()}")
            #     print(f"Current Loss: {L}")

        return model(text_X).view(-1)



=======
        # write your solution here
        pass
>>>>>>> 9016849dd7c10148d239af7ddbf3f8f0c09c2cb0

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class HW(object):
    def __init__(self, list_of_tasks: List[Task]):
        self.tasks = list_of_tasks
        self.hw_name = "dl_lesson_1_checker_hw"

    def evaluate(self):
        aggregated_tasks = []

        for task in self.tasks:
            aggregated_tasks.append(task.evaluate())

        aggregated_tasks = {"tasks": aggregated_tasks}

        yaml_result = yaml.dump(aggregated_tasks)

        print(yaml_result)

        with open(f"{self.hw_name}.yaml", "w") as f:
            f.write(yaml_result)


hw = HW([Task1(), Task2(), Task3()])
hw.evaluate()
