# Creates a Firedrake defcon container

FROM firedrakeproject/firedrake-vanilla-default:latest

# This DockerFile is looked after by
MAINTAINER Jack Betteridge <j.betteridge@imperial.ac.uk>

# Install defcon into /opt (installing build dependencies first)
RUN pip install setuptools \
    && pip install --verbose --src /opt \
        --editable git+https://bitbucket.org/pefarrell/defcon.git#egg=defcon
