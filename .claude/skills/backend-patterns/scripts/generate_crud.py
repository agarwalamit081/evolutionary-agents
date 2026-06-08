"""Generate service class boilerplate with CRUD methods.
Usage: python generate_crud.py --entity User --lang python [--pattern simple|repository]
       python generate_crud.py --entity Product --lang typescript
"""
import argparse


def generate_python_simple(entity: str) -> str:
    e = entity.lower()
    return f"""# {entity} Service

class {entity}Service:
    def __init__(self, repository):
        self.repo = repository

    async def get_{e}(self, {e}_id: str) -> dict | None:
        return await self.repo.get_by_id({e}_id)

    async def create_{e}(self, data: dict) -> dict:
        # TODO: Add validation and business logic
        return await self.repo.create(data)

    async def update_{e}(self, {e}_id: str, data: dict) -> dict:
        # TODO: Add business logic
        return await self.repo.update({e}_id, data)

    async def delete_{e}(self, {e}_id: str) -> bool:
        # TODO: Consider soft delete
        return await self.repo.delete({e}_id)
"""


def generate_python_repository(entity: str) -> str:
    e = entity.lower()
    tbl = e + "s"
    return (
        f"# {entity} Repository Interface and Implementation\n\n"
        "from abc import ABC, abstractmethod\n\n\n"
        f"class {entity}Repository(ABC):\n"
        f"    @abstractmethod\n"
        f"    async def get_by_id(self, {e}_id: str) -> dict | None: ...\n"
        f"    @abstractmethod\n"
        f"    async def create(self, data: dict) -> dict: ...\n"
        f"    @abstractmethod\n"
        f"    async def update(self, {e}_id: str, data: dict) -> dict: ...\n"
        f"    @abstractmethod\n"
        f"    async def delete(self, {e}_id: str) -> bool: ...\n\n\n"
        f"class Postgres{entity}Repository({entity}Repository):\n"
        f"    def __init__(self, db):\n"
        f"        self.db = db\n\n"
        f"    async def get_by_id(self, {e}_id: str) -> dict | None:\n"
        f'        row = await self.db.fetchone("SELECT * FROM {tbl} WHERE id = $1", {e}_id)\n'
        f"        return dict(row) if row else None\n\n"
        f"    async def create(self, data: dict) -> dict:\n"
        f'        cols = ", ".join(data.keys())\n'
        f'        vals = ", ".join(f"${{i+1}}" for i in range(len(data)))\n'
        f"        row = await self.db.fetchone(\n"
        f'            f"INSERT INTO {tbl} ({{cols}}) VALUES ({{vals}}) RETURNING *",\n'
        f"            *data.values()\n"
        f"        )\n"
        f"        return dict(row)\n\n"
        f"    async def update(self, {e}_id: str, data: dict) -> dict:\n"
        f'        sets = ", ".join(f"{{k}} = ${{i+2}}" for i, k in enumerate(data.keys()))\n'
        f"        row = await self.db.fetchone(\n"
        f'            f"UPDATE {tbl} SET {{sets}} WHERE id = $1 RETURNING *",\n'
        f"            {e}_id, *data.values()\n"
        f"        )\n"
        f"        return dict(row)\n\n"
        f"    async def delete(self, {e}_id: str) -> bool:\n"
        f'        result = await self.db.execute("DELETE FROM {tbl} WHERE id = $1", {e}_id)\n'
        f'        return result == "DELETE 1"\n'
    )


def generate_typescript(entity: str) -> str:
    return f"""// {entity} Service

import {{ Repository }} from '../lib/repository';

export interface {entity}Data {{
  // TODO: Define fields
}}

export class {entity}Service {{
  constructor(private repo: Repository<{entity}Data>) {{}}

  async get(id: string): Promise<{entity}Data | null> {{
    return this.repo.findById(id);
  }}

  async create(data: {entity}Data): Promise<{entity}Data> {{
    // TODO: Add validation
    return this.repo.create(data);
  }}

  async update(id: string, data: Partial<{entity}Data>): Promise<{entity}Data> {{
    return this.repo.update(id, data);
  }}

  async delete(id: string): Promise<boolean> {{
    return this.repo.delete(id);
  }}
}}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate CRUD service boilerplate")
    parser.add_argument("--entity", required=True, help="Entity name (e.g., User, Product)")
    parser.add_argument("--lang", choices=["python", "typescript"], default="python")
    parser.add_argument("--pattern", choices=["simple", "repository"], default="simple",
                        help="Python only: simple service or repository pattern")
    args = parser.parse_args()

    if args.lang == "typescript":
        code = generate_typescript(args.entity)
    elif args.pattern == "repository":
        code = generate_python_repository(args.entity)
    else:
        code = generate_python_simple(args.entity)

    print(code)


if __name__ == "__main__":
    main()
