-- access_by_lua_file: runs FIRST, captures the POST body and rewrites it through the internal Perl
-- /fixup location (lib/perl/llama.pm) before it is sent off to the proxy
if ngx.req.get_method() == "POST" then
  -- 1. Read request body into memory
  ngx.req.read_body()
  local req_body = ngx.req.get_body_data()

  -- 2. Pass body to internal Perl subrequest
  local r = ngx.location.capture("/fixup", {
      method = ngx.HTTP_POST,
      body = req_body,
      always_forward_body = true
  })

  -- 3. Check subrequest execution & set updated body
  if r.status == ngx.HTTP_OK and r.body then
      ngx.req.set_body_data(r.body)
      ngx.req.set_header("Content-Length", string.len(r.body))
  end
end
