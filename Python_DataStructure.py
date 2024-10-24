from typing import Any, Optional, List, Dict, TypeVar
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, TypeVar, Generic
from collections import deque
import heapq
from bidi.algorithm import get_display
from arabic_reshaper import reshape
T = TypeVar('T')

class LimitedArray:
    """یک آرایه با سایز محدود که امکان اضافه کردن آیتم تا حد مشخص را فراهم می‌کند.
    
    Args:
        max_size (int): حداکثر تعداد آیتم‌های مجاز
    
    Examples:
        >>> arr = LimitedArray(3)
        >>> arr.add(1)
        >>> arr.add(2)
        >>> print(arr)
        [1, 2]
    """
    
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._array: List[Any] = []
        
    @property
    def is_full(self) -> bool:
        """بررسی پر بودن آرایه"""
        return len(self._array) >= self._max_size
        
    @property
    def is_empty(self) -> bool:
        """بررسی خالی بودن آرایه"""
        return len(self._array) == 0

    def add(self, item: Any) -> None:
        """افزودن آیتم به آرایه
        
        Raises:
            OverflowError: اگر آرایه پر باشد
        """
        if not self.is_full:
            self._array.append(item)
        else:
            raise OverflowError(get_display(reshape(f"آرایه نمی‌تواند بیش از {self._max_size} آیتم داشته باشد.")))

    def clear(self) -> None:
        """پاک کردن تمام آیتم‌های آرایه"""
        self._array.clear()

    def __getitem__(self, index: int) -> Any:
        return self._array[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._array[index] = value

    def __len__(self) -> int:
        return len(self._array)

    def __repr__(self) -> str:
        return repr(self._array)


class Stack(LimitedArray):
    """پیاده‌سازی ساختار داده پشته (LIFO) با استفاده از آرایه محدود
    
    Examples:
        >>> stack = Stack(3)
        >>> stack.push(1)
        >>> stack.push(2)
        >>> stack.pop()
        2
    """
    
    def push(self, item: Any) -> None:
        """افزودن آیتم به بالای پشته"""
        self.add(item)

    def pop(self) -> Any:
        """برداشتن و برگرداندن آیتم از بالای پشته
        
        Raises:
            IndexError: اگر پشته خالی باشد
        """
        if not self.is_empty:
            return self._array.pop()
        raise IndexError(get_display(reshape("پشته خالی است")))

    def peek(self) -> Any:
        """مشاهده آیتم بالای پشته بدون برداشتن آن
        
        Raises:
            IndexError: اگر پشته خالی باشد
        """
        if not self.is_empty:
            return self._array[-1]
        raise IndexError(get_display(reshape("پشته خالی است")))


class Queue(LimitedArray):
    """پیاده‌سازی صف (FIFO) با استفاده از آرایه محدود و deque برای کارایی بهتر
    
    Examples:
        >>> queue = Queue(3)
        >>> queue.enqueue(1)
        >>> queue.enqueue(2)
        >>> queue.dequeue()
        1
    """
    
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._array = deque(maxlen=max_size)

    def enqueue(self, item: Any) -> None:
        """افزودن آیتم به انتهای صف"""
        self.add(item)

    def dequeue(self) -> Any:
        """برداشتن و برگرداندن آیتم از ابتدای صف
        
        Raises:
            IndexError: اگر صف خالی باشد
        """
        if not self.is_empty:
            return self._array.popleft()
        raise IndexError(get_display(reshape("صف خالی است")))

    def peek(self) -> Any:
        """مشاهده آیتم ابتدای صف بدون برداشتن آن
        
        Raises:
            IndexError: اگر صف خالی باشد
        """
        if not self.is_empty:
            return self._array[0]
        raise IndexError(get_display(reshape("صف خالی است")))


class Node(Generic[T]):
    """گره پایه برای ساختارهای داده مبتنی بر پیوند
    
    Args:
        data: داده ذخیره شده در گره
    """
    
    def __init__(self, data: T):
        self.data: T = data
        self.next: Optional[Node[T]] = None


class LinkedList(Generic[T]):
    """پیاده‌سازی لیست پیوندی یک طرفه
    
    Examples:
        >>> ll = LinkedList[int]()
        >>> ll.append(1)
        >>> ll.append(2)
        >>> print(ll)
        1 -> 2
    """
    
    def __init__(self):
        self.head: Optional[Node[T]] = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """افزودن داده به انتهای لیست"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def prepend(self, data: T) -> None:
        """افزودن داده به ابتدای لیست"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def delete(self, data: T) -> bool:
        """حذف اولین گره با داده مشخص شده
        
        Returns:
            bool: True اگر داده پیدا و حذف شد، False در غیر این صورت
        """
        if not self.head:
            return False

        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        return False

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        nodes = []
        current = self.head
        while current:
            nodes.append(repr(current.data))
            current = current.next
        return " -> ".join(nodes)


class CircularLinkedList(LinkedList[T]):
    """پیاده‌سازی لیست پیوندی حلقوی
    
    Examples:
        >>> cll = CircularLinkedList[str]()
        >>> cll.append("A")
        >>> cll.append("B")
        >>> print(cll)
        'A' -> 'B' -> 'A'
    """
    
    def append(self, data: T) -> None:
        """افزودن داده به انتهای لیست حلقوی"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
        self._size += 1

    def delete(self, data: T) -> bool:
        """حذف اولین گره با داده مشخص شده
        
        Returns:
            bool: True اگر داده پیدا و حذف شد، False در غیر این صورت
        """
        if not self.head:
            return False

        if self.head.data == data:
            if self._size == 1:
                self.head = None
            else:
                current = self.head
                while current.next != self.head:
                    current = current.next
                current.next = self.head.next
                self.head = self.head.next
            self._size -= 1
            return True

        current = self.head
        while current.next != self.head:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        return False

    def __repr__(self) -> str:
        if not self.head:
            return "Empty list"
        
        nodes = []
        current = self.head
        while True:
            nodes.append(repr(current.data))
            current = current.next
            if current == self.head:
                break
        return " -> ".join(nodes)


class TreeNode(Generic[T]):
    """گره درخت دودویی
    
    Args:
        data: داده ذخیره شده در گره
    """
    
    def __init__(self, data: T):
        self.data: T = data
        self.left: Optional[TreeNode[T]] = None
        self.right: Optional[TreeNode[T]] = None


class BinaryTree(Generic[T]):
    """پیاده‌سازی درخت دودویی با پیمایش‌های مختلف
    
    Examples:
        >>> tree = BinaryTree(1)
        >>> tree.add_left(tree.root, 2)
        >>> tree.add_right(tree.root, 3)
        >>> print(tree.inorder_traversal())
        [2, 1, 3]
    """
    
    def __init__(self, root_data: T):
        self.root = TreeNode(root_data)

    def add_left(self, parent_node: TreeNode[T], data: T) -> None:
        """افزودن فرزند چپ به گره والد"""
        if not parent_node.left:
            parent_node.left = TreeNode(data)
        else:
            raise ValueError(get_display(reshape("فرزند چپ از قبل وجود دارد")))

    def add_right(self, parent_node: TreeNode[T], data: T) -> None:
        """افزودن فرزند راست به گره والد"""
        if not parent_node.right:
            parent_node.right = TreeNode(data)
        else:
            raise ValueError(get_display(reshape("فرزند راست از قبل وجود دارد")))

    def inorder_traversal(self) -> List[T]:
        """پیمایش درون‌ترتیبی (چپ، ریشه، راست)"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[TreeNode[T]], result: List[T]) -> None:
        if node:
            self._inorder(node.left, result)
            result.append(node.data)
            self._inorder(node.right, result)

    def preorder_traversal(self) -> List[T]:
        """پیمایش پیش‌ترتیبی (ریشه، چپ، راست)"""
        result = []
        self._preorder(self.root, result)
        return result

    def _preorder(self, node: Optional[TreeNode[T]], result: List[T]) -> None:
        if node:
            result.append(node.data)
            self._preorder(node.left, result)
            self._preorder(node.right, result)

    def postorder_traversal(self) -> List[T]:
        """پیمایش پس‌ترتیبی (چپ، راست، ریشه)"""
        result = []
        self._postorder(self.root, result)
        return result

    def _postorder(self, node: Optional[TreeNode[T]], result: List[T]) -> None:
        if node:
            self._postorder(node.left, result)
            self._postorder(node.right, result)
            result.append(node.data)


class Graph:
    """پیاده‌سازی گراف با استفاده از لیست مجاورت
    
    Examples:
        >>> g = Graph()
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 3)
        >>> print(g.bfs(1))
        [1, 2, 3]
    """
    
    def __init__(self):
        self.graph: Dict[Any, List[Any]] = {}

    def add_vertex(self, vertex: Any) -> None:
        """افزودن رأس به گراف"""
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, u: Any, v: Any) -> None:
        """افزودن یال بین دو رأس"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.graph[u].append(v)
        self.graph[v].append(u)  # برای گراف بدون جهت

    def bfs(self, start: Any) -> List[Any]:
        """جستجوی سطح به سطح از رأس شروع
        
        Args:
            start: رأس شروع
            
        Returns:
            List[Any]: لیست رئوس در ترتیب ملاقات
        """
        visited = set()
        queue = deque([start])
        result = []

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend(v for v in self.graph[vertex] if v not in visited)

        return result

    def dfs(self, start: Any) -> List[Any]:
        """جستجوی عمق اول از رأس شروع
        
        Args:
            start: رأس شروع
            
        Returns:
            List[Any]: لیست رئوس در ترتیب ملاقات
        """
        visited = set()
        result = []

        def _dfs_recursive(vertex: Any) -> None:
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    _dfs_recursive(neighbor)

        _dfs_recursive(start)
        return result

    def __repr__(self) -> str:
        return '\n'.join([f"{vertex}: {edges}" for vertex, edges in self.graph.items()])


class PriorityQueue(Generic[T]):
    """پیاده‌سازی صف اولویت با استفاده از هیپ
    
    Examples:
        >>> pq = PriorityQueue[int]()
        >>> pq.push(3)
        >>> pq.push(1)
        >>> pq.push(2)
        >>> pq.pop()
        1
    """
    
    def __init__(self):
        self._heap: List[T] = []

    def push(self, item: T) -> None:
        """افزودن آیتم به صف اولویت"""
        heapq.heappush(self._heap, item)

    def pop(self) -> T:
        """برداشتن و برگرداندن آیتم با کمترین مقدار
        
        Raises:
            IndexError: اگر صف خالی باشد
        """
        if self._heap:
            return heapq.heappop(self._heap)
        raise IndexError(get_display(reshape("صف اولویت خالی است")))

    def peek(self) -> T:
        """مشاهده آیتم با کمترین مقدار بدون برداشتن آن
        
        Raises:
            IndexError: اگر صف خالی باشد
        """
        if self._heap:
            return self._heap[0]
        raise IndexError(get_display(reshape("صف اولویت خالی است")))

    def __len__(self) -> int:
        return len(self._heap)


class AVLNode(Generic[T]):
    """گره درخت AVL
    
    Args:
        data: داده ذخیره شده در گره
    """
    
    def __init__(self, data: T):
        self.data: T = data
        self.left: Optional[AVLNode[T]] = None
        self.right: Optional[AVLNode[T]] = None
        self.height: int = 1


class AVLTree(Generic[T]):
    """پیاده‌سازی درخت AVL با قابلیت متعادل‌سازی خودکار
    
    Examples:
        >>> avl = AVLTree[int]()
        >>> avl.insert(1)
        >>> avl.insert(2)
        >>> avl.insert(3)  # چرخش خودکار برای متعادل‌سازی
        >>> print(avl.inorder_traversal())
        [1, 2, 3]
    """
    
    def __init__(self):
        self.root: Optional[AVLNode[T]] = None

    def height(self, node: Optional[AVLNode[T]]) -> int:
        """برگرداندن ارتفاع گره"""
        if not node:
            return 0
        return node.height

    def balance_factor(self, node: Optional[AVLNode[T]]) -> int:
        """محاسبه فاکتور توازن گره"""
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)

    def update_height(self, node: AVLNode[T]) -> None:
        """به‌روزرسانی ارتفاع گره"""
        node.height = max(self.height(node.left), self.height(node.right)) + 1

    def right_rotate(self, y: AVLNode[T]) -> AVLNode[T]:
        """چرخش راست برای متعادل‌سازی"""
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self.update_height(y)
        self.update_height(x)

        return x

    def left_rotate(self, x: AVLNode[T]) -> AVLNode[T]:
        """چرخش چپ برای متعادل‌سازی"""
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self.update_height(x)
        self.update_height(y)

        return y

    def insert(self, data: T) -> None:
        """درج داده در درخت با حفظ توازن"""
        self.root = self._insert_recursive(self.root, data)

    def _insert_recursive(self, node: Optional[AVLNode[T]], data: T) -> AVLNode[T]:
        if not node:
            return AVLNode(data)

        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            node.right = self._insert_recursive(node.right, data)
        else:
            return node

        self.update_height(node)

        balance = self.balance_factor(node)

        # حالت‌های عدم توازن و چرخش
        if balance > 1:
            if data < node.left.data:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)

        if balance < -1:
            if data > node.right.data:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)

        return node

    def inorder_traversal(self) -> List[T]:
        """پیمایش درون‌ترتیبی درخت"""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: Optional[AVLNode[T]], result: List[T]) -> None:
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)


class Trie:
    """پیاده‌سازی درخت تِرای برای ذخیره و جستجوی رشته‌ها
    
    Examples:
        >>> trie = Trie()
        >>> trie.insert("hello")
        >>> trie.search("hello")
        True
        >>> trie.starts_with("hel")
        True
    """
    
    class TrieNode:
        def __init__(self):
            self.children: Dict[str, Trie.TrieNode] = {}
            self.is_end_of_word: bool = False

    def __init__(self):
        self.root = self.TrieNode()

    def insert(self, word: str) -> None:
        """درج کلمه در درخت"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """جستجوی کلمه در درخت
        
        Returns:
            bool: True اگر کلمه دقیقاً در درخت وجود داشته باشد
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """بررسی وجود کلمه با پیشوند مشخص
        
        Returns:
            bool: True اگر کلمه‌ای با پیشوند مشخص شده وجود داشته باشد
        """
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """یافتن گره متناظر با پیشوند"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


class DisjointSet:
    """پیاده‌سازی ساختار داده مجموعه‌های مجزا با فشرده‌سازی مسیر
    
    Examples:
        >>> ds = DisjointSet()
        >>> ds.make_set(1)
        >>> ds.make_set(2)
        >>> ds.union(1, 2)
        >>> ds.find(1) == ds.find(2)
        True
    """
    
    def __init__(self):
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def make_set(self, x: Any) -> None:
        """ایجاد مجموعه تک عضوی"""
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x: Any) -> Any:
        """یافتن نماینده مجموعه با فشرده‌سازی مسیر"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Any, y: Any) -> None:
        """ادغام دو مجموعه با اتصال بر اساس رتبه"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

class SortingAlgorithms:
    """مجموعه الگوریتم‌های مرتب‌سازی
    
    Examples:
        >>> sorter = SortingAlgorithms()
        >>> arr = [64, 34, 25, 12, 22, 11, 90]
        >>> print(sorter.quick_sort(arr.copy()))
        [11, 12, 22, 25, 34, 64, 90]
    """

    @staticmethod
    def bubble_sort(arr: List[T]) -> List[T]:
        """مرتب‌سازی حبابی"""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    @staticmethod
    def quick_sort(arr: List[T]) -> List[T]:
        """مرتب‌سازی سریع"""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)

    @staticmethod
    def merge_sort(arr: List[T]) -> List[T]:
        """مرتب‌سازی ادغامی"""
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])

        return SortingAlgorithms._merge(left, right)

    @staticmethod
    def _merge(left: List[T], right: List[T]) -> List[T]:
        """ادغام دو لیست مرتب شده"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    @staticmethod
    def heap_sort(arr: List[T]) -> List[T]:
        """مرتب‌سازی هرمی"""
        def heapify(arr: List[T], n: int, i: int) -> None:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)

        return arr

