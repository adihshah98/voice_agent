import { NextRequest, NextResponse } from "next/server";

const PUBLIC_PATHS = ["/login"];

export default function proxy(req: NextRequest) {
  const path = req.nextUrl.pathname;
  const isPublic = PUBLIC_PATHS.some((p) => path.startsWith(p));

  const token = req.cookies.get("access_token")?.value;

  if (!isPublic && !token) {
    return NextResponse.redirect(new URL("/login", req.nextUrl));
  }

  if (isPublic && token) {
    return NextResponse.redirect(new URL("/", req.nextUrl));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon\\.ico).*)"],
};
