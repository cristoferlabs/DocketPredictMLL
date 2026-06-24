#!/usr/bin/env bash
# Railway start command for ARQ worker service
exec arq apps.worker.main.WorkerSettings
