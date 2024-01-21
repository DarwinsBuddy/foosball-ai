import os.path

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

from foosball import main, get_argparse

WEBHOOK_ADDRESS = "localhost"
WEBHOOK_PORT = 8080

TABLE_ID = "123"
BLUE = "BLUE"
RED = "RED"

GOAL_REQUEST = {
    "method": "POST",
    "uri": "/v1/api/goal"
}

main_args = {
    "webhook": True,
    "headless": True,
    "ball": "yellow"
}


class TestContext:
    __test__ = False
    goals: [str]

    def __init__(self):
        self.goals = []

    def count_goal(self, team: str):
        self.goals.append(team)


@pytest.fixture()
def test_context():
    print("setup test_context")
    yield TestContext()
    print("teardown test_context")


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return WEBHOOK_ADDRESS, WEBHOOK_PORT


def get_handler(ctx: TestContext):
    def handle(request) -> Response:
        print("RECEIVED : ", request.json)
        assert 'tableId' in request.json
        assert 'team' in request.json
        ctx.count_goal(request.json['team'])

        return Response({})

    return handle


@pytest.mark.asyncio
class TestGoals:
    @staticmethod
    async def launch_ai(path_to_video_file: str) -> None:
        assert os.path.isfile(os.path.abspath(path_to_video_file))

        args = {**vars(get_argparse().parse_args()), **main_args, "file": path_to_video_file}
        assert main(args) is None

    async def test_goal(self, httpserver: HTTPServer, test_context: TestContext):
        (httpserver
         .expect_oneshot_request(**GOAL_REQUEST)
         .respond_with_handler(get_handler(test_context))
         )
        await self.launch_ai(os.path.abspath("tests/assets/goal5.mp4"))
        assert test_context.goals == [BLUE]
