from arbitrium_core.domain.workflow.nodes.base import (
    BaseNode,
    ExecutionContext,
    Port,
    PortType,
)


def register_all() -> None:
    import arbitrium_core.domain.workflow.nodes.evaluation
    import arbitrium_core.domain.workflow.nodes.flow
    import arbitrium_core.domain.workflow.nodes.generation
    import arbitrium_core.domain.workflow.nodes.input
    import arbitrium_core.domain.workflow.nodes.knowledge
    import arbitrium_core.domain.workflow.nodes.llm
    import arbitrium_core.domain.workflow.nodes.output

    del (
        arbitrium_core.domain.workflow.nodes.evaluation,
        arbitrium_core.domain.workflow.nodes.flow,
        arbitrium_core.domain.workflow.nodes.generation,
        arbitrium_core.domain.workflow.nodes.input,
        arbitrium_core.domain.workflow.nodes.knowledge,
        arbitrium_core.domain.workflow.nodes.llm,
        arbitrium_core.domain.workflow.nodes.output,
    )


__all__ = ["BaseNode", "ExecutionContext", "Port", "PortType", "register_all"]
