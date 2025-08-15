// Main JavaScript functionality for the Churn Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize any charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Form validation enhancement
    enhanceFormValidation();
    
    // Initialize feature animations
    initializeAnimations();
});

// Enhanced form validation
function enhanceFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Focus on first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                }
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Real-time validation feedback
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                this.classList.remove('is-valid');
                this.classList.add('is-invalid');
            }
        });
    });
}

// Animation initialization
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements with fade-in-up class
    document.querySelectorAll('.fade-in-up').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s ease';
        observer.observe(el);
    });
}

// Chart initialization (if Chart.js is available)
function initializeCharts() {
    // Feature importance chart
    const featureChart = document.getElementById('featureImportanceChart');
    if (featureChart) {
        const ctx = featureChart.getContext('2d');
        // Chart implementation would go here
    }

    // Risk distribution chart
    const riskChart = document.getElementById('riskDistributionChart');
    if (riskChart) {
        const ctx = riskChart.getContext('2d');
        // Chart implementation would go here
    }
}

// Utility functions
class ChurnPredictor {
    constructor() {
        this.apiEndpoint = '/api/predict';
        this.model = null;
    }

    // Predict churn probability
    async predictChurn(customerData) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(customerData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    // Validate customer data
    validateCustomerData(data) {
        const required = [
            'tenure', 'monthly_charges', 'total_charges',
            'contract', 'payment_method', 'internet_service'
        ];

        for (let field of required) {
            if (!data[field] && data[field] !== 0) {
                return {
                    valid: false,
                    error: `Missing required field: ${field}`
                };
            }
        }

        // Validate numeric ranges
        if (data.tenure < 0 || data.tenure > 100) {
            return {
                valid: false,
                error: 'Tenure must be between 0 and 100 months'
            };
        }

        if (data.monthly_charges < 0 || data.monthly_charges > 200) {
            return {
                valid: false,
                error: 'Monthly charges must be between $0 and $200'
            };
        }

        return { valid: true };
    }

    // Format currency
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    // Format percentage
    formatPercentage(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 1
        }).format(value);
    }

    // Get risk level from probability
    getRiskLevel(probability) {
        if (probability < 0.3) return 'low';
        if (probability < 0.7) return 'medium';
        return 'high';
    }

    // Get risk color
    getRiskColor(riskLevel) {
        const colors = {
            low: '#198754',    // success green
            medium: '#ffc107', // warning yellow
            high: '#dc3545'    // danger red
        };
        return colors[riskLevel] || '#6c757d';
    }
}

// Global instance
const churnPredictor = new ChurnPredictor();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChurnPredictor };
}

// Progress indicator for long forms
function updateProgressIndicator() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const inputs = form.querySelectorAll('input[required], select[required]');
    const filled = Array.from(inputs).filter(input => input.value.trim() !== '').length;
    const progress = (filled / inputs.length) * 100;

    const progressBar = document.querySelector('.form-progress');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
}

// Auto-save form data to localStorage
function autoSaveFormData() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const inputs = form.querySelectorAll('input, select');
    const formData = {};

    inputs.forEach(input => {
        formData[input.name] = input.value;
    });

    localStorage.setItem('churnPredictionFormData', JSON.stringify(formData));
}

// Restore form data from localStorage
function restoreFormData() {
    const savedData = localStorage.getItem('churnPredictionFormData');
    if (!savedData) return;

    try {
        const formData = JSON.parse(savedData);
        Object.keys(formData).forEach(key => {
            const input = document.querySelector(`[name="${key}"]`);
            if (input && formData[key]) {
                input.value = formData[key];
            }
        });
    } catch (error) {
        console.error('Error restoring form data:', error);
    }
}

// Clear saved form data
function clearSavedFormData() {
    localStorage.removeItem('churnPredictionFormData');
}

// Add event listeners for auto-save
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    if (form) {
        // Restore data on page load
        restoreFormData();

        // Auto-save on input change
        form.addEventListener('input', function() {
            autoSaveFormData();
            updateProgressIndicator();
        });

        // Clear saved data on successful submission
        form.addEventListener('submit', function() {
            clearSavedFormData();
        });
    }
});

// Dark mode toggle (if implemented)
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
}

// Initialize dark mode from saved preference
function initializeDarkMode() {
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
        document.body.classList.add('dark-mode');
    }
}

// Call on page load
document.addEventListener('DOMContentLoaded', initializeDarkMode);
