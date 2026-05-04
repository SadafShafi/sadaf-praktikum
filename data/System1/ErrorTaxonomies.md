# Error Taxonomies

## 1. Resource Exhaustion (Memory/OOM)
* **Trigger:** `ansible_memfree_mb` is only 145, and `ansible_swapfree_mb` is 0.
* **Error Generated:** `Out of Memory (OOM)` kills, `Killed` signal terminated program, or JVM initialization failures.

## 2. Storage Exhaustion
* **Trigger:** `size_available` on `/` is approximately 120MB.
* **Error Generated:** `No space left on device` or `Disk quota exceeded` when trying to download large Docker images, build large binaries, or write heavy logs.

## 3. Process Lock & Daemon Conflict
* **Trigger:** `active_locks` lists `/var/lib/dpkg/lock-frontend`.
* **Error Generated:** `Could not get lock /var/lib/dpkg/lock-frontend` when the agent attempts to use `apt` or `dpkg` to install missing dependencies.

## 4. Network Port Collision
* **Trigger:** `bound_tcp_ports` shows ports 80, 8080, and 5432 are actively listening.
* **Error Generated:** `bind: address already in use` or `listen tcp4 0.0.0.0:8080` failures when trying to start a new web application, proxy, or database.

## 5. Privilege & Access Control
* **Trigger:** The active `USER` is `deploy_user`, but `critical_file_permissions` enforce `root:root` ownership without group write access (e.g., 0644 or 0755).
* **Error Generated:** `Permission denied` when the agent tries to blindly echo text into `/etc/app/config.yml` or manipulate logs in `/var/log/app/` without using `sudo` or proper user escalation.
