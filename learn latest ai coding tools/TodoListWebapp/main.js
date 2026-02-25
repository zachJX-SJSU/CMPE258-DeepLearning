class TodoStore {
    constructor() {
        this.tasks = JSON.parse(localStorage.getItem('calendarTasks')) || {};
    }

    getTasks(dateKey) {
        return this.tasks[dateKey] || [];
    }

    addTask(dateKey, text) {
        if (!this.tasks[dateKey]) this.tasks[dateKey] = [];
        this.tasks[dateKey].push({ id: Date.now(), text, completed: false });
        this.save();
    }

    deleteTask(dateKey, id) {
        this.tasks[dateKey] = this.tasks[dateKey].filter(t => t.id !== id);
        this.save();
    }

    save() {
        localStorage.setItem('calendarTasks', JSON.stringify(this.tasks));
    }
}

const store = new TodoStore();
let currentViewDate = new Date();
let selectedDateKey = null;

function renderCalendar() {
    const grid = document.getElementById('calendar-grid');
    const monthDisplay = document.getElementById('month-display');
    grid.innerHTML = '';

    const year = currentViewDate.getFullYear();
    const month = currentViewDate.getMonth();
    
    monthDisplay.innerText = new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric' }).format(currentViewDate);

    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    for (let i = 0; i < firstDay; i++) {
        grid.innerHTML += `<div></div>`;
    }

    for (let day = 1; day <= daysInMonth; day++) {
        const dateKey = `${year}-${month + 1}-${day}`;
        const dayCell = document.createElement('div');
        dayCell.className = `day-cell ${dateKey === selectedDateKey ? 'selected' : ''}`;
        if (store.getTasks(dateKey).length > 0) dayCell.classList.add('has-tasks');
        
        dayCell.innerHTML = `<span>${day}</span>`;
        dayCell.onclick = () => handleDaySelection(dateKey);
        grid.appendChild(dayCell);
    }
}

function handleDaySelection(dateKey) {
    selectedDateKey = dateKey;
    document.getElementById('selected-date-title').innerText = dateKey;
    renderTodoList();
    renderCalendar(); // Refresh selection visual
}

function renderTodoList() {
    const list = document.getElementById('todo-list');
    const tasks = store.getTasks(selectedDateKey);
    list.innerHTML = '';
    
    document.getElementById('todo-count').innerText = `${tasks.length} tasks`;

    tasks.forEach(task => {
        const li = document.createElement('li');
        li.className = 'todo-item';
        li.innerHTML = `
            <span>${task.text}</span>
            <button onclick="removeTask(${task.id})" style="background:transparent; padding:0">✕</button>
        `;
        list.appendChild(li);
    });
}

// Global exposure for the simple HTML onclick
window.removeTask = (id) => {
    store.deleteTask(selectedDateKey, id);
    renderTodoList();
    renderCalendar();
};

document.getElementById('add-todo-btn').onclick = () => {
    const input = document.getElementById('todo-input');
    if (!input.value || !selectedDateKey) return;
    store.addTask(selectedDateKey, input.value);
    input.value = '';
    renderTodoList();
    renderCalendar();
};

document.getElementById('prev-month').onclick = () => { currentViewDate.setMonth(currentViewDate.getMonth() - 1); renderCalendar(); };
document.getElementById('next-month').onclick = () => { currentViewDate.setMonth(currentViewDate.getMonth() + 1); renderCalendar(); };

renderCalendar();