#!/bin/bash
# Quick management commands for the GCP GPU VM
# Usage: ./gcp-manage.sh [start|stop|status|ssh|logs|destroy|restart-server]

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-quantization-gpu}"

case "${1:-help}" in
  start)
    echo "Starting VM $INSTANCE_NAME..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID"
    sleep 10
    # Restart the server after boot
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      cd ~/quantization-app/backend
      pkill -f 'uvicorn main:app' 2>/dev/null || true
      sleep 2
      export CORS_ORIGINS='*'
      nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > ~/backend.log 2>&1 &
      sleep 3
      curl -s http://localhost:8000/device-info
    "
    EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo ""
    echo "VM running at: http://$EXTERNAL_IP:8000"
    ;;

  stop)
    echo "Stopping VM $INSTANCE_NAME (stops billing)..."
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID"
    echo "VM stopped. Disk charges still apply (~$5/month for 100GB SSD)."
    ;;

  status)
    gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
      --format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP,machineType.basename())"
    ;;

  ssh)
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID"
    ;;

  logs)
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
      --command="tail -100 ~/backend.log"
    ;;

  restart-server)
    echo "Restarting backend server..."
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      cd ~/quantization-app/backend
      pkill -f 'uvicorn main:app' 2>/dev/null || true
      sleep 2
      export CORS_ORIGINS='*'
      nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > ~/backend.log 2>&1 &
      sleep 3
      curl -s http://localhost:8000/health
    "
    ;;

  upload)
    echo "Uploading latest backend code..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    gcloud compute scp --recurse \
      "$SCRIPT_DIR/../backend" \
      "$INSTANCE_NAME":~/quantization-app/ \
      --zone="$ZONE" --project="$PROJECT_ID"
    echo "Code uploaded. Run './gcp-manage.sh restart-server' to apply."
    ;;

  destroy)
    echo "WARNING: This will permanently delete the VM and its disk."
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet
      echo "VM deleted."
    fi
    ;;

  *)
    echo "GCP GPU VM Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start          - Start the VM and backend server"
    echo "  stop           - Stop the VM (saves money, keeps disk)"
    echo "  status         - Show VM status and IP"
    echo "  ssh            - SSH into the VM"
    echo "  logs           - View backend server logs"
    echo "  restart-server - Restart the backend process"
    echo "  upload         - Upload latest backend code to VM"
    echo "  destroy        - Permanently delete the VM"
    echo ""
    echo "Environment variables:"
    echo "  GCP_PROJECT_ID    - GCP project (default: gcloud config)"
    echo "  GCP_ZONE          - Compute zone (default: us-central1-a)"
    echo "  GCP_INSTANCE_NAME - VM name (default: quantization-gpu)"
    ;;
esac
