import { readFileSync } from "node:fs";

const html = readFileSync("app/static/index.html", "utf8");
const matches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];

if (matches.length === 0) {
  throw new Error("No inline script found in app/static/index.html");
}

for (const [index, match] of matches.entries()) {
  try {
    new Function(match[1]);
  } catch (error) {
    throw new Error(`Inline demo script ${index + 1} has invalid JavaScript: ${error.message}`);
  }
}

console.log(`Demo JavaScript syntax passed (${matches.length} inline script block)`);
