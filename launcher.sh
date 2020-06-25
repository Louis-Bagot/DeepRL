DOCKER_IMAGE="gitlab.ilabt.imec.be:4567/lbagot/explore_option:dqn_breakout"
GPULAB_PROJECT="explore_option"

# docker login
docker login --username $GITLAB_IDLAB_USERNAME --password $GITLAB_IDLAB_PASSWORD gitlab.ilabt.imec.be:4567

# build & push the docker-image
docker build -t $DOCKER_IMAGE .
docker push $DOCKER_IMAGE

# submit the job
gpulab-cli submit --project=${GPULAB_PROJECT} < gpulab_job.js
