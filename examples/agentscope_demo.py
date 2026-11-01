"""Offline AgentScope message/single/fixed-multi-agent contrast in an isolated v1 env."""
import asyncio

import agentscope
from agentscope.agent import AgentBase
from agentscope.message import Msg


class OfflineRoleAgent(AgentBase):
    def __init__(self, name: str, action: str) -> None:
        super().__init__()
        self.name, self.action = name, action

    async def reply(self, message: Msg) -> Msg:
        return Msg(self.name, f"{message.get_text_content()} -> {self.action}", "assistant")


async def main() -> None:
    start = Msg("user", "study AgentOS", "user")
    single = await OfflineRoleAgent("single", "research+analyze+verify")(start)
    researcher = await OfflineRoleAgent("researcher", "research")(start)
    analyst = await OfflineRoleAgent("analyst", "analyze")(researcher)
    verifier = await OfflineRoleAgent("verifier", "verify")(analyst)
    print({"agentscope_version": agentscope.__version__, "single": single.get_text_content(), "fixed_multi": verifier.get_text_content()})


if __name__ == "__main__":
    asyncio.run(main())

