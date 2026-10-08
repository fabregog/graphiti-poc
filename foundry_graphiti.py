"""Graphiti wired to Azure AI Foundry (LLM + embeddings) and Neo4j."""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient

load_dotenv()


class FoundryReasoningLLMClient(OpenAIClient):
    """OpenAIClient for Foundry reasoning deployments such as gpt-6-luna.

    graphiti-core 0.30.2 only treats deployments named o1*/o3*/gpt-5* as reasoning
    models. For any other name it sends `temperature` / `max_tokens` and no reasoning
    effort, which gpt-6-luna rejects (it requires `max_completion_tokens` and does not
    support the `minimal` effort graphiti falls back to). This client always sends an
    explicit, supported reasoning effort.

    AzureOpenAILLMClient is not used on purpose: in 0.30.2 its
    `_handle_structured_response` returns a dict while BaseOpenAIClient unpacks a
    (response, input_tokens, output_tokens) tuple.
    """

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        return await self.client.responses.parse(
            model=model,
            input=messages,  # type: ignore[arg-type]
            max_output_tokens=max_tokens,
            text_format=response_model,
            reasoning={'effort': reasoning},  # type: ignore[typeddict-item]
        )

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            response_format={'type': 'json_object'},
            reasoning_effort=self.reasoning,  # type: ignore[arg-type]
        )


def build_graphiti() -> Graphiti:
    foundry = AsyncOpenAI(
        base_url=os.environ['AZURE_OPENAI_BASE_URL'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
    )
    llm_deployment = os.environ['AZURE_LLM_DEPLOYMENT']

    # Edge dedup/contradiction runs on the "small" model, so both must point at the deployment.
    llm_client = FoundryReasoningLLMClient(
        config=LLMConfig(model=llm_deployment, small_model=llm_deployment),
        client=foundry,
        reasoning=os.getenv('LLM_REASONING_EFFORT', 'low'),
    )
    embedder = AzureOpenAIEmbedderClient(
        foundry, model=os.environ['AZURE_EMBEDDING_DEPLOYMENT']
    )
    # add_triplet() searches with RRF and never calls the reranker; passing one bound to
    # Foundry only stops Graphiti from building a default client against public OpenAI.
    cross_encoder = OpenAIRerankerClient(config=LLMConfig(model=llm_deployment), client=foundry)

    return Graphiti(
        os.environ['NEO4J_URI'],
        os.environ['NEO4J_USER'],
        os.environ['NEO4J_PASSWORD'],
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
