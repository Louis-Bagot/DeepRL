{
  "jobDefinition": {
    "name": "dqn_breakout",
    "dockerImage": "gitlab+deploy-token-197:pVdPsj5atFfJS55NnvpC@gitlab.ilabt.imec.be:4567/lbagot/explore_option:dqn_breakout",
    "clusterId": 7,
    "command": "",
    "resources": {
      "gpus": 1,
      "systemMemory": 64000,
      "cpuCores": 4,
      "minCudaVersion": 10
    },
  "jobDataLocations":[
                {
                   "mountPoint": "/app/",
                   "sharePath": "/project/"
                }
      ]
  }
}
