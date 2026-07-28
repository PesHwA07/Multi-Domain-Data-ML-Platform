# AWS Cloud Deployment (Day 27)

This directory contains the necessary configurations to deploy the Multi-Domain ML Platform to AWS Free Tier (EC2 + RDS).

## Architecture for Production

In our local setup, PostgreSQL was hosted as a Docker container. For production resilience, we decouple the database and host it on managed **Amazon RDS**. The applications (Airflow, FastAPI, Streamlit) will run on a single **Amazon EC2** instance via `docker-compose`.

## 1. Setup Amazon RDS (PostgreSQL)
1. Go to the AWS RDS Console and create a new **PostgreSQL 15** database.
2. Select the **Free Tier** template (db.t3.micro).
3. Ensure the database is placed in a public subnet or accessible by your EC2 instance's Security Group.
4. Note your endpoint URL, username, and password. 
5. Construct your `DATABASE_URL`: 
   `postgresql+psycopg2://<user>:<password>@<rds-endpoint>:5432/<db_name>`

## 2. Setup Amazon EC2
1. Launch an **Ubuntu Server 22.04 LTS** EC2 instance (t2.micro for Free Tier, though t3.medium is recommended due to Airflow memory requirements).
2. Configure the Security Group to allow inbound traffic on:
   - Port 22 (SSH)
   - Port 8080 (Airflow)
   - Port 8000 (FastAPI)
   - Port 8501 (Streamlit)
3. SSH into your instance and install Docker & Docker Compose.

## 3. Deployment Steps

1. **Clone the repository** onto your EC2 instance.
2. **Build and push images** to Amazon ECR (or build them directly on the EC2 instance to save registry costs).
3. **Set Environment Variables**:
   ```bash
   export AWS_RDS_DATABASE_URL="postgresql+psycopg2://..."
   export AIRFLOW_FERNET_KEY="..."
   export ECR_REGISTRY="your-ecr-registry-url"
   ```
4. **Deploy**:
   ```bash
   docker-compose -f deploy/docker-compose.prod.yml up -d
   ```
