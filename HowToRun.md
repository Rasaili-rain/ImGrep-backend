 # Image Search Backend - Project Overview

 ## What This Is
 A Django REST API backend running in Docker with PostgreSQL database. Pure API service designed to work with any frontend (React, Vue, mobile apps, etc.).

 ## Prerequisites
 - Windows 10/11 with Docker Desktop
 - Docker Desktop: [Download](https://www.docker.com/products/docker-desktop/)
 - Text Editor: VS Code recommended

 ## Quick Setup

 1. **Install Docker Desktop**
    ```bash
    # Test installation
    docker --version
    ```

 2. **Create Project**
    ```bash
    mkdir image-search-backend
    cd image-search-backend
    ```

 3. **Copy Project Files**
    - Copy all files from the setup guide into your project folder.

 4. **Start Project**
    ```bash
    # Start all services
    docker-compose up --build

    # In another terminal, setup database
    docker-compose exec web python manage.py migrate
    ```

 5. **Test API**
    - Health check: http://localhost:8000/api/health/
    - Test endpoint: http://localhost:8000/api/test/

 ## Project Structure
 ```
 image-search-backend/
 ├── docker-compose.yml     # Docker services config
 ├── Dockerfile            # Python environment setup
 ├── requirements.txt      # Python packages
 ├── .env                 # Environment variables
 ├── manage.py            # Django management
 └── backend/
     ├── settings.py      # Django configuration
     ├── urls.py          # Main URL routing
     ├── wsgi.py          # WSGI application
     └── api/
         ├── views.py     # API endpoint logic
         └── urls.py      # API URL routing
 ```

 ## Daily Development Workflow

 ### Starting Work
 ```bash
 # Start all services
 docker-compose up

 # Or run in background
 docker-compose up -d
 ```

 ### Making Changes
 1. **Code Changes**
    - Edit files in `backend/api/views.py` for API logic
    - Edit files in `backend/api/urls.py` for new endpoints
    - Changes are automatically reloaded (no restart needed)

 2. **Database Changes**
    ```bash
    # Create migration after model changes
    docker-compose exec web python manage.py makemigrations

    # Apply migrations
    docker-compose exec web python manage.py migrate
    ```

 3. **Adding New Packages**
    - Add package to requirements.txt
    - Rebuild: `docker-compose up --build`

 ### Testing Changes
 ```bash
 # Test API endpoints
 curl http://localhost:8000/api/health/

 # View logs
 docker-compose logs web

 # Django shell for debugging
 docker-compose exec web python manage.py shell
 ```

 ### Stopping Work
 ```bash
 # Stop services
 docker-compose down

 # Stop and remove all data
 docker-compose down -v
 ```

 ## Common Development Tasks
 ### Add New API Endpoint
 1. Add view function in `backend/api/views.py`:
    ```python
    @api_view(['GET'])
    def my_new_endpoint(request):
        return Response({'message': 'Hello from new endpoint'})
    ```
    
 2. Add URL in `backend/api/urls.py`:
    ```python
    path('my-endpoint/', views.my_new_endpoint),
    ```
    
 3. Test: http://localhost:8000/api/my-endpoint/

 ### Database Models
 - Create models in new file `backend/api/models.py`
 - Add to settings: Add 'backend.api' to INSTALLED_APPS (already done)
 - Create migration: `docker-compose exec web python manage.py makemigrations`
 - Apply migration: `docker-compose exec web python manage.py migrate`

 ### Environment Variables
 - Edit `.env` file for configuration
 - Restart services: `docker-compose down && docker-compose up`

 ## API Endpoints
 ### Current Endpoints
 - GET /api/health/ - Health check
 - GET /api/test/ - Test GET request
 - POST /api/test/ - Test POST request

 ### Testing with cURL
 ```bash
 # Health check
 curl http://localhost:8000/api/health/

 # GET test
 curl http://localhost:8000/api/test/

 # POST test
 curl -X POST http://localhost:8000/api/test/ \
   -H "Content-Type: application/json" \
   -d '{"name": "test", "data": "hello world"}'
 ```

 ## Troubleshooting
 ### Common Issues
 ```bash
 # Port already in use
 docker-compose down
 docker-compose up

 # Database connection issues
 docker-compose exec web python manage.py migrate

 # Package not found
 docker-compose up --build

 # Clean restart
 docker-compose down -v
 docker-compose up --build
 ```

 ### Useful Commands
 ```bash
 # View running containers
 docker ps

 # View all logs
 docker-compose logs

 # View specific service logs
 docker-compose logs web
 docker-compose logs db

 # Access container shell
 docker-compose exec web bash

 # Database shell
 docker-compose exec db psql -U postgres -d imagedb
 ```

 ## Next Steps
 ### For Image Search Features
 - Add image upload endpoint
 - Add image processing models
 - Add search endpoints
 - Add vector similarity search

 ### For Production
 - Add authentication
 - Add rate limiting
 - Add proper error handling
 - Add API documentation
 - Add tests

 ## File Locations
 ### Add API Logic
 - Views: `backend/api/views.py`
 - URLs: `backend/api/urls.py`
 - Models: `backend/api/models.py` (create if needed)

 ### Configuration
 - Django Settings: `backend/settings.py`
 - Database: `docker-compose.yml`
 - Packages: `requirements.txt`

 # Ready to develop! The backend provides a clean REST API foundation that any frontend can consume.