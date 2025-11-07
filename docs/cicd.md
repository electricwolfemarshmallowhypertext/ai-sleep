# Check Statuses and Conclusions

Checks can have many different statuses.  
Statuses describe the state of a check from when it is created to when it is completed.  
Some statuses cannot be set manually and are reserved for GitHub Actions.  
When a check has a status of **completed**, it has a **conclusion**.  
The conclusion describes the result of the check.

## Statuses

| Status | Description | GitHub Actions only? |
|---------|--------------|----------------------|
| completed | The check run completed and has a conclusion (see below). | No |
| expected | The check run is waiting for a status to be reported. | Yes |
| failure | The check run failed. | No |
| in_progress | The check run is in progress. | No |
| pending | The check run is at the front of the queue but the group-based concurrency limit has been reached. | Yes |
| queued | The check run has been queued. | No |
| requested | The check run has been created but has not been queued. | Yes |
| startup_failure | The check suite failed during startup. This status is not applicable to check runs. | Yes |
| waiting | The check run is waiting for a deployment protection rule to be satisfied. | Yes |

## Conclusions

| Conclusion | Description |
|-------------|-------------|
| action_required | The check run provided required actions upon its completion. |
| cancelled | The check run was cancelled before it completed. |
| failure | The check run failed. |
| neutral | The check run completed with a neutral result. This is treated as a success for dependent checks in GitHub Actions. |
| skipped | The check run was skipped. This is treated as a success for dependent checks in GitHub Actions. |
| stale | The check run was marked stale by GitHub because it took too long. |
| success | The check run completed successfully. |
| timed_out | The check run timed out. |
