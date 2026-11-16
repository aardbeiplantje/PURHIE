group "default" {
  targets = ["local"]
  default = true
}
group "release" {
  targets = ["containers"]
}
group "local" {
  targets = ["_local"]
}
variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_REPOSITORY" {
  default = "ai"
}
variable "DOCKER_IMAGE_NAME" {
  default = "purhie"
}
variable "LEMONADE_LLAMACPP_VERSION" {
  default = "b9586"
}
variable "DOCKER_TAG" {
  default = "latest"
}
variable "GFX_VERSION" {
  default = "-gfx1151"
}
variable "CACHEBUST" {
  default = "2026-11-14-10"
}
target "_common" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  networks = ["host"]
  buildkit = true
  target = "runtime"
  args = {
    CACHEBUST = "${CACHEBUST}"
    LEMONADE_LLAMACPP_VERSION = "${LEMONADE_LLAMACPP_VERSION}"
  }
  progress = ["plain", "tty"]
}

target "_local" {
  inherits = ["_common"]
  tags = [
    "local/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}",
  ]
  output = [
    "type=docker,name=local/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}"
  ]
}

target "containers" {
  inherits = ["_common"]
  pull = true
  name = "containers-${env}"
  matrix = {
    env = ["release"]
  }
  tags = [
    "${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}",
  ]
  output = [
    "type=image,name=${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG},push=true"
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}
